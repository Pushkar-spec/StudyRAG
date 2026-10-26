"""Local vector storage and semantic/hybrid retrieval for StudyRAG.

Manages SentenceTransformers embedding generation, local ChromaDB vector indexing,
incremental SQLite-ChromaDB synchronization, semantic candidate retrieval,
and unified hybrid search combining FTS5 lexical matching and semantic vector similarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
import sqlite3
from typing import Any, Callable

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None  # type: ignore[assignment]
    ChromaSettings = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment]

from core.config import Settings, get_chroma_dir, load
from core.db import (
    db_session,
    delete_file_vector_state,
    get_chunk_with_metadata,
    get_pending_chunks_for_indexing,
    get_vector_index_counts,
    record_chunks_indexed_batch,
    search_chunks_fts,
    update_file_indexing_status,
)

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "studyrag_chunks"
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass
class IndexingResult:
    """Summary of a vector indexing run."""

    total_eligible: int = 0
    indexed_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class SemanticSearchResult:
    """A single result from semantic vector search."""

    chunk_id: str
    file_id: int
    filename: str
    group_id: int
    text: str
    location: str
    page: int | None
    semantic_distance: float
    semantic_similarity: float
    sender: str | None = None


@dataclass
class HybridSearchResult:
    """A single deduplicated result from unified hybrid search."""

    chunk_id: str
    file_id: int
    filename: str
    group_id: int
    text: str
    location: str
    page: int | None
    matched_by: str  # 'keyword', 'semantic', or 'both'
    keyword_score: float | None = None
    semantic_distance: float | None = None
    semantic_similarity: float | None = None
    sender: str | None = None


# ---------------------------------------------------------------------------
# Embedding Model Manager
# ---------------------------------------------------------------------------

def compute_chunk_content_hash(text: str) -> str:
    """Calculate SHA-256 hash of chunk text for vector cache invalidation."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class LocalEmbeddingManager:
    """Singleton manager for local SentenceTransformer models with lazy loading."""

    _instance: LocalEmbeddingManager | None = None
    _cached_model: Any = None
    _cached_model_name: str | None = None

    @classmethod
    def get_instance(cls) -> LocalEmbeddingManager:
        """Return the shared singleton manager instance."""
        if cls._instance is None:
            cls._instance = LocalEmbeddingManager()
        return cls._instance

    def get_model(self, model_name: str = DEFAULT_EMBED_MODEL) -> Any:
        """Retrieve or lazily initialize the SentenceTransformer model."""
        if SentenceTransformer is None:
            raise ImportError(
                "The 'sentence-transformers' package is not installed. "
                "Install it using 'pip install sentence-transformers'."
            )

        if self._cached_model is not None and self._cached_model_name == model_name:
            return self._cached_model

        logger.info("Loading local embedding model: %s", model_name)
        try:
            model = SentenceTransformer(model_name)
            self._cached_model = model
            self._cached_model_name = model_name
            return model
        except Exception as err:
            logger.error("Failed to load embedding model '%s': %s", model_name, err)
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}': {err}"
            ) from err

    def embed_texts(
        self,
        texts: list[str],
        model_name: str = DEFAULT_EMBED_MODEL,
        custom_embedder: Callable[[list[str]], list[list[float]]] | None = None,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Generate normalized vector embeddings for a list of strings."""
        if not texts:
            return []

        if custom_embedder is not None:
            return custom_embedder(texts)

        model = self.get_model(model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [e.tolist() for e in embeddings]

    def embed_query(
        self,
        query: str,
        model_name: str = DEFAULT_EMBED_MODEL,
        custom_embedder: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> list[float]:
        """Generate normalized vector embedding for a single search query."""
        results = self.embed_texts([query], model_name=model_name, custom_embedder=custom_embedder)
        return results[0]


# ---------------------------------------------------------------------------
# ChromaDB Client & Collection Management
# ---------------------------------------------------------------------------

def get_chroma_client(chroma_dir: Path | str | None = None) -> Any:
    """Create or return a persistent ChromaDB client pointing to local storage."""
    if chromadb is None:
        raise ImportError(
            "The 'chromadb' package is not installed. "
            "Install it using 'pip install chromadb'."
        )

    resolved_path = Path(chroma_dir).resolve() if chroma_dir else get_chroma_dir()
    resolved_path.mkdir(parents=True, exist_ok=True)

    settings = ChromaSettings(
        is_persistent=True,
        persist_directory=str(resolved_path),
        anonymized_telemetry=False,
    )
    return chromadb.PersistentClient(
        path=str(resolved_path),
        settings=settings,
    )


def get_or_create_collection(
    client: Any | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chroma_dir: Path | str | None = None,
) -> Any:
    """Retrieve or initialize the ChromaDB persistent collection with cosine space."""
    ch_client = client or get_chroma_client(chroma_dir)
    return ch_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Incremental Indexing & Synchronization
# ---------------------------------------------------------------------------

def index_pending_chunks(
    group_id: int | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embed_model: str = DEFAULT_EMBED_MODEL,
    chroma_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    custom_embedder: Callable[[list[str]], list[list[float]]] | None = None,
    batch_size: int = 32,
) -> IndexingResult:
    """Incrementally generate embeddings and index pending chunks into ChromaDB.

    Idempotent: skips chunks whose content hash and model version are already indexed.

    Args:
        group_id: Optional group filter. If None, indexes across all study groups.
        collection_name: Target ChromaDB collection name.
        embed_model: Configured model identifier.
        chroma_dir: Target ChromaDB storage path.
        db_path: Target SQLite database path.
        custom_embedder: Optional custom embedding callable for testing.
        batch_size: Chunk batch size for embedding inference and upsert.

    Returns:
        IndexingResult with execution statistics.
    """
    res = IndexingResult()

    try:
        collection = get_or_create_collection(
            collection_name=collection_name,
            chroma_dir=chroma_dir,
        )
    except Exception as err:
        msg = f"Failed to initialize ChromaDB collection: {err}"
        logger.error(msg)
        res.errors.append(msg)
        return res

    pending_rows = get_pending_chunks_for_indexing(
        group_id=group_id,
        embed_model=embed_model,
        db_path=db_path,
    )

    # Filter out chunks whose current content hash matches the indexed hash
    chunks_to_embed: list[sqlite3.Row] = []
    for row in pending_rows:
        res.total_eligible += 1
        current_hash = compute_chunk_content_hash(row["text"])
        if row["current_indexed_hash"] == current_hash:
            res.skipped_count += 1
        else:
            chunks_to_embed.append(row)

    if not chunks_to_embed:
        # Check and update file indexing status flags in SQLite
        affected_file_ids = {int(r["file_id"]) for r in pending_rows}
        for fid in affected_file_ids:
            update_file_indexing_status(fid, embed_model=embed_model, db_path=db_path)
        return res

    embed_mgr = LocalEmbeddingManager.get_instance()

    # Process in batches
    for i in range(0, len(chunks_to_embed), batch_size):
        batch = chunks_to_embed[i : i + batch_size]
        batch_texts = [str(r["text"]) for r in batch]
        batch_ids = [str(r["id"]) for r in batch]
        batch_hashes = [compute_chunk_content_hash(t) for t in batch_texts]

        try:
            embeddings = embed_mgr.embed_texts(
                batch_texts,
                model_name=embed_model,
                custom_embedder=custom_embedder,
                batch_size=batch_size,
            )

            metadatas = [
                {
                    "chunk_id": str(r["id"]),
                    "file_id": int(r["file_id"]),
                    "group_id": int(r["group_id"]),
                    "page": int(r["page"]) if r["page"] is not None else -1,
                    "location": str(r["location"] or ""),
                    "chunk_index": int(r["chunk_index"]) if r["chunk_index"] is not None else 0,
                    "content_hash": batch_hashes[idx],
                    "embed_model": embed_model,
                }
                for idx, r in enumerate(batch)
            ]

            # Upsert into ChromaDB
            collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=batch_texts,
            )

            # Persist state records in SQLite
            db_records = [
                (batch_ids[idx], int(batch[idx]["file_id"]), embed_model, batch_hashes[idx])
                for idx in range(len(batch))
            ]
            record_chunks_indexed_batch(db_records, db_path=db_path)

            res.indexed_count += len(batch)

        except Exception as err:
            err_msg = f"Failed to index batch {i // batch_size + 1}: {err}"
            logger.error(err_msg)
            res.failed_count += len(batch)
            res.errors.append(err_msg)

    # Update file indexing flags
    all_file_ids = {int(r["file_id"]) for r in chunks_to_embed}
    for fid in all_file_ids:
        update_file_indexing_status(fid, embed_model=embed_model, db_path=db_path)

    return res


# ---------------------------------------------------------------------------
# Semantic Search
# ---------------------------------------------------------------------------

def search_chunks_semantic(
    query: str,
    top_k: int = 10,
    group_id: int | None = None,
    file_id: int | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embed_model: str = DEFAULT_EMBED_MODEL,
    chroma_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    custom_embedder: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[SemanticSearchResult]:
    """Execute local semantic vector search against ChromaDB resolved against SQLite.

    Args:
        query: Search topic or natural language phrase.
        top_k: Maximum candidate chunks to retrieve (> 0).
        group_id: Optional study group filter.
        file_id: Optional file filter.
        collection_name: Target ChromaDB collection.
        embed_model: Configured model identifier.
        chroma_dir: Optional custom ChromaDB directory.
        db_path: Optional custom SQLite database path.
        custom_embedder: Optional custom embedding function for testing.

    Returns:
        List of SemanticSearchResult instances sorted by cosine distance ascending.
    """
    clean_query = query.strip()
    if not clean_query:
        return []
    if top_k <= 0:
        raise ValueError(f"top_k must be greater than 0, got {top_k}")

    collection = get_or_create_collection(
        collection_name=collection_name,
        chroma_dir=chroma_dir,
    )

    if collection.count() == 0:
        return []

    embed_mgr = LocalEmbeddingManager.get_instance()
    query_vector = embed_mgr.embed_query(
        clean_query,
        model_name=embed_model,
        custom_embedder=custom_embedder,
    )

    where_filter: dict[str, Any] | None = None
    if group_id is not None and file_id is not None:
        where_filter = {"$and": [{"group_id": int(group_id)}, {"file_id": int(file_id)}]}
    elif group_id is not None:
        where_filter = {"group_id": int(group_id)}
    elif file_id is not None:
        where_filter = {"file_id": int(file_id)}

    query_limit = min(top_k, max(1, collection.count()))

    try:
        raw_results = collection.query(
            query_embeddings=[query_vector],
            n_results=query_limit,
            where=where_filter,
            include=["metadatas", "distances"],
        )
    except Exception as err:
        logger.warning("ChromaDB query failed: %s", err)
        return []

    results: list[SemanticSearchResult] = []
    if not raw_results or not raw_results.get("ids") or not raw_results["ids"][0]:
        return []

    ids = raw_results["ids"][0]
    distances = raw_results["distances"][0] if "distances" in raw_results and raw_results["distances"] else [0.0] * len(ids)

    for chunk_id, dist in zip(ids, distances):
        # Resolve authoritative record from SQLite
        db_row = get_chunk_with_metadata(chunk_id, db_path=db_path)
        if not db_row:
            # Stale vector whose chunk was deleted from SQLite
            continue

        cos_dist = float(dist)
        cos_sim = max(0.0, 1.0 - cos_dist)

        results.append(
            SemanticSearchResult(
                chunk_id=chunk_id,
                file_id=int(db_row["file_id"]),
                filename=str(db_row["filename"]),
                group_id=int(db_row["group_id"]),
                text=str(db_row["text"]),
                location=str(db_row["location"] or ""),
                page=int(db_row["page"]) if db_row["page"] is not None else None,
                semantic_distance=cos_dist,
                semantic_similarity=cos_sim,
                sender=str(db_row["sender"]) if db_row["sender"] else None,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Hybrid Retrieval (FTS5 + Semantic)
# ---------------------------------------------------------------------------

def search_chunks_hybrid(
    query: str,
    limit: int = 20,
    group_id: int | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embed_model: str = DEFAULT_EMBED_MODEL,
    chroma_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    settings: Settings | None = None,
    custom_embedder: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[HybridSearchResult]:
    """Execute unified hybrid retrieval combining FTS5 lexical matching and semantic vector search.

    Deduplicates candidates on chunk ID, records whether each hit was discovered by
    'keyword', 'semantic', or 'both', and preserves separate search scores.

    Args:
        query: User search query or syllabus topic.
        limit: Maximum total candidate chunks to return.
        group_id: Optional study group constraint.
        collection_name: Target ChromaDB collection name.
        embed_model: Embedding model name.
        chroma_dir: Optional custom ChromaDB directory.
        db_path: Optional custom SQLite database path.
        settings: Optional custom Settings.
        custom_embedder: Optional custom embedding function for testing.

    Returns:
        List of HybridSearchResult instances.
    """
    clean_query = query.strip()
    if not clean_query:
        return []
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    # 1. Lexical Keyword Retrieval (FTS5)
    keyword_rows = search_chunks_fts(
        query=clean_query,
        group_id=group_id,
        limit=limit,
        db_path=db_path,
    )

    # 2. Semantic Vector Retrieval (ChromaDB)
    semantic_results: list[SemanticSearchResult] = []
    try:
        semantic_results = search_chunks_semantic(
            query=clean_query,
            top_k=limit,
            group_id=group_id,
            collection_name=collection_name,
            embed_model=embed_model,
            chroma_dir=chroma_dir,
            db_path=db_path,
            custom_embedder=custom_embedder,
        )
    except Exception as err:
        logger.warning(
            "Semantic retrieval unavailable; falling back to keyword search only: %s",
            err,
        )

    # 3. Deduplicate and Merge Candidate Results
    merged_map: dict[str, HybridSearchResult] = {}

    # Process keyword matches
    for rank_idx, row in enumerate(keyword_rows, start=1):
        c_id = str(row["id"])
        # Inverse rank score (1.0 for first hit, decaying)
        kw_score = 1.0 / (1.0 + 0.1 * (rank_idx - 1))
        merged_map[c_id] = HybridSearchResult(
            chunk_id=c_id,
            file_id=int(row["file_id"]),
            filename=str(row["filename"]),
            group_id=int(row["group_id"]),
            text=str(row["text"]),
            location=str(row["location"] or ""),
            page=int(row["page"]) if row["page"] is not None else None,
            matched_by="keyword",
            keyword_score=kw_score,
            sender=str(row["sender"]) if "sender" in row.keys() and row["sender"] else None,
        )

    # Process semantic matches
    for s_res in semantic_results:
        c_id = s_res.chunk_id
        if c_id in merged_map:
            # Overlapping candidate matched by both search methods
            existing = merged_map[c_id]
            existing.matched_by = "both"
            existing.semantic_distance = s_res.semantic_distance
            existing.semantic_similarity = s_res.semantic_similarity
        else:
            merged_map[c_id] = HybridSearchResult(
                chunk_id=c_id,
                file_id=s_res.file_id,
                filename=s_res.filename,
                group_id=s_res.group_id,
                text=s_res.text,
                location=s_res.location,
                page=s_res.page,
                matched_by="semantic",
                semantic_distance=s_res.semantic_distance,
                semantic_similarity=s_res.semantic_similarity,
                sender=s_res.sender,
            )

    # 4. Sort results
    # Candidates matched by 'both' appear first as consensus hits,
    # followed by semantic similarity and keyword rank.
    def candidate_sort_key(res: HybridSearchResult) -> tuple[int, float]:
        method_priority = 2 if res.matched_by == "both" else 1
        sem_score = res.semantic_similarity if res.semantic_similarity is not None else 0.0
        kw_score = res.keyword_score if res.keyword_score is not None else 0.0
        return (method_priority, max(sem_score, kw_score))

    sorted_results = sorted(merged_map.values(), key=candidate_sort_key, reverse=True)
    return sorted_results[:limit]
