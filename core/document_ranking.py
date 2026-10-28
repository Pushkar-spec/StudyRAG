"""Document-level ranking and end-to-end search orchestration for StudyRAG.

Groups hybrid and cross-encoder reranked chunks by parent document, aggregates
relevance signals with a bounded non-inflationary formula, and orchestrates
the complete retrieval-rerank-ranking pipeline with resilient fallbacks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
from typing import Any, Callable

from core.config import Settings, load
from core.db import db_session
from core.reranker import (
    DEFAULT_RERANK_MODEL,
    RerankedChunk,
    rerank_chunks,
)
from core.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBED_MODEL,
    HybridSearchResult,
    search_chunks_hybrid,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RankedDocument",
    "SearchRankedResult",
    "rank_documents_from_chunks",
    "search_ranked_documents",
]


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass
class RankedDocument:
    """A ranked original document with supporting evidence and relevance breakdown."""

    file_id: int
    filename: str
    doc_type: str
    group_id: int
    group_name: str
    document_score: float
    rank: int
    supporting_chunks: list[Any] = field(default_factory=list)
    source_locations: list[str] = field(default_factory=list)
    excerpts: list[str] = field(default_factory=list)
    chunk_count: int = 0
    keyword_matches_count: int = 0
    semantic_matches_count: int = 0
    consensus_matches_count: int = 0
    is_reranked: bool = True
    diagnostics: list[str] = field(default_factory=list)
    whatsapp_context: Any | None = None
    content_score: float | None = None
    content_rank: int | None = None
    context_score: Any | None = None
    adjustment_explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass instance to a standard dictionary."""
        d = asdict(self)
        if self.whatsapp_context and hasattr(self.whatsapp_context, "to_dict"):
            d["whatsapp_context"] = self.whatsapp_context.to_dict()
        if self.context_score and hasattr(self.context_score, "to_dict"):
            d["context_score"] = self.context_score.to_dict()
        return d


@dataclass
class SearchRankedResult:
    """Full outcome of an end-to-end search and ranking query."""

    query: str
    documents: list[RankedDocument]
    chunks: list[Any]
    total_candidates: int
    is_reranked: bool
    retrieval_mode: str  # 'hybrid_reranked', 'hybrid_fallback', 'keyword_only', 'empty'
    diagnostics: list[str] = field(default_factory=list)
    applied_filters: dict[str, Any] | None = None
    excluded_documents_count: int = 0
    is_context_ranked: bool = False
    context_adjustments_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "documents": [d.to_dict() for d in self.documents],
            "chunks": [c.to_dict() if hasattr(c, "to_dict") else dict(c) for c in self.chunks],
            "total_candidates": self.total_candidates,
            "is_reranked": self.is_reranked,
            "retrieval_mode": self.retrieval_mode,
            "diagnostics": self.diagnostics,
            "applied_filters": self.applied_filters,
            "excluded_documents_count": self.excluded_documents_count,
            "is_context_ranked": self.is_context_ranked,
            "context_adjustments_count": self.context_adjustments_count,
        }


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _is_near_duplicate(text1: str, text2: str, threshold: float = 0.60) -> bool:
    """Check if two chunk texts share high token overlap (e.g. sliding window overlap)."""
    import re
    words1 = set(re.findall(r"\w+", text1.lower()))
    words2 = set(re.findall(r"\w+", text2.lower()))
    if not words1 or not words2:
        return False
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return (len(intersection) / len(union)) >= threshold


def _generate_excerpt(text: str, max_chars: int = 180) -> str:
    """Generate a clean, single-line excerpt from chunk text."""
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def _resolve_document_metadata(
    file_id: int,
    db_path: Path | str | None = None,
) -> tuple[str, str, int, str]:
    """Retrieve authoritative (filename, ext, group_id, group_name) from SQLite."""
    try:
        with db_session(db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT f.filename, f.ext, f.group_id, g.name as group_name
                FROM files f
                LEFT JOIN groups g ON f.group_id = g.id
                WHERE f.id = ?;
                """,
                (file_id,),
            )
            row = cur.fetchone()
            if row:
                return (
                    str(row["filename"] or f"file_{file_id}"),
                    str(row["ext"] or ""),
                    int(row["group_id"] or 0),
                    str(row["group_name"] or f"Group #{row['group_id']}"),
                )
    except Exception as err:
        logger.debug("Failed to resolve metadata for file_id %d from SQLite: %s", file_id, err)
    return (f"file_{file_id}", "", 0, "Unknown Group")


def _get_chunk_score(chunk: Any) -> float:
    """Extract or derive the primary score for a chunk."""
    if hasattr(chunk, "rerank_score") and chunk.rerank_score is not None:
        return float(chunk.rerank_score)
    if isinstance(chunk, dict) and "rerank_score" in chunk and chunk["rerank_score"] is not None:
        return float(chunk["rerank_score"])

    # Fallback score derivation when cross-encoder is not used
    matched_by = getattr(chunk, "matched_by", None) or (chunk.get("matched_by") if isinstance(chunk, dict) else "")
    sem_sim = getattr(chunk, "semantic_similarity", None) or (chunk.get("semantic_similarity") if isinstance(chunk, dict) else None)
    kw_score = getattr(chunk, "keyword_score", None) or (chunk.get("keyword_score") if isinstance(chunk, dict) else None)

    sim = float(sem_sim) if sem_sim is not None else 0.0
    kw = float(kw_score) if kw_score is not None else 0.0

    if matched_by == "both":
        return 1.0 + max(sim, kw) * 0.5
    if matched_by == "semantic":
        return sim
    if matched_by == "keyword":
        return kw
    return max(sim, kw)


# ---------------------------------------------------------------------------
# Document-Level Aggregation
# ---------------------------------------------------------------------------

def rank_documents_from_chunks(
    chunks: list[Any],
    group_id: int | None = None,
    db_path: Path | str | None = None,
    top_k_docs: int | None = None,
) -> list[RankedDocument]:
    """Group candidate chunks by parent document and compute bounded relevance scores.

    Aggregation Strategy:
    1. Primary signal: The strongest chunk's score (s1).
    2. Multi-chunk evidence bonus (bounded): Up to 3 additional distinct, non-redundant
       chunks contribute with diminishing weights (w2=0.10, w3=0.05, w4=0.02).
       Near-duplicate chunks (>70% token overlap) are filtered out to prevent score inflation.
    3. Document score = s1 + sum(wi * max(0.0, si)).
       A document cannot dominate simply by possessing many weak or repetitive chunks.

    Args:
        chunks: List of RerankedChunk or HybridSearchResult objects.
        group_id: Optional study group filter.
        db_path: Optional custom SQLite database path.
        top_k_docs: Optional limit on the number of returned ranked documents.

    Returns:
        List of RankedDocument instances ordered by document_score descending.
    """
    if not chunks:
        return []

    # 1. Group chunks by file_id
    grouped: dict[int, list[Any]] = {}
    for c in chunks:
        fid = getattr(c, "file_id", None)
        if fid is None and isinstance(c, dict):
            fid = c.get("file_id")
        if fid is None:
            continue
        grouped.setdefault(int(fid), []).append(c)

    if not grouped:
        return []

    ranked_docs: list[RankedDocument] = []

    for fid, file_chunks in grouped.items():
        # Sort chunks for this file by descending score
        file_chunks.sort(key=lambda c: _get_chunk_score(c), reverse=True)

        is_reranked = any(
            hasattr(c, "rerank_score") and c.rerank_score is not None
            for c in file_chunks
        )

        # Authoritative metadata lookup
        db_filename, db_ext, db_gid, db_gname = _resolve_document_metadata(fid, db_path=db_path)

        # Fallback to chunk attributes if DB metadata is placeholder
        c0 = file_chunks[0]
        c0_filename = getattr(c0, "filename", None) or (c0.get("filename") if isinstance(c0, dict) else "")
        c0_gid = getattr(c0, "group_id", None) or (c0.get("group_id") if isinstance(c0, dict) else None)

        filename = db_filename if db_filename and not db_filename.startswith("file_") else (c0_filename or f"file_{fid}")
        doc_type = db_ext if db_ext else (Path(filename).suffix.lower() if filename else "")
        resolved_gid = db_gid if db_gid > 0 else (int(c0_gid) if c0_gid else 0)
        group_name = db_gname if db_gname != "Unknown Group" else f"Group #{resolved_gid}"

        # Bounded score aggregation
        s1 = _get_chunk_score(file_chunks[0])
        doc_score = s1

        # Track distinct non-redundant chunks for bonus
        distinct_bonus_chunks: list[tuple[float, Any]] = []
        accepted_texts: list[str] = [
            getattr(file_chunks[0], "text", "") or (file_chunks[0].get("text", "") if isinstance(file_chunks[0], dict) else "")
        ]

        decay_weights = [0.10, 0.05, 0.02]

        for cand in file_chunks[1:]:
            cand_text = getattr(cand, "text", "") or (cand.get("text", "") if isinstance(cand, dict) else "")
            cand_score = _get_chunk_score(cand)

            # Check if cand is a near-duplicate of an already accepted chunk
            if any(_is_near_duplicate(cand_text, prev_t) for prev_t in accepted_texts):
                continue

            accepted_texts.append(cand_text)
            distinct_bonus_chunks.append((cand_score, cand))
            if len(distinct_bonus_chunks) >= len(decay_weights):
                break

        bonus_total = 0.0
        for weight, (b_score, _) in zip(decay_weights, distinct_bonus_chunks):
            # Only positive chunk scores contribute to bonus
            if b_score > 0.0:
                delta = weight * b_score
                bonus_total += delta

        doc_score += bonus_total

        # Collect source locations, excerpts, and match statistics
        locations_set: list[str] = []
        excerpts: list[str] = []
        kw_count = 0
        sem_count = 0
        consensus_count = 0

        for c in file_chunks:
            loc = getattr(c, "location", None) or (c.get("location") if isinstance(c, dict) else "")
            page = getattr(c, "page", None) or (c.get("page") if isinstance(c, dict) else None)
            matched_by = getattr(c, "matched_by", None) or (c.get("matched_by") if isinstance(c, dict) else "")
            text = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else "")

            clean_loc = loc.strip() if loc else (f"Page {page}" if page is not None else "")
            if clean_loc and clean_loc not in locations_set:
                locations_set.append(clean_loc)

            if len(excerpts) < 3 and text:
                excerpts.append(_generate_excerpt(text))

            if matched_by == "both":
                consensus_count += 1
            elif matched_by == "semantic":
                sem_count += 1
            elif matched_by == "keyword":
                kw_count += 1

        diagnostics: list[str] = []
        if is_reranked:
            diagnostics.append(f"Reranked with cross-encoder. Primary chunk score: {s1:.4f}.")
            if bonus_total > 0:
                diagnostics.append(
                    f"Multi-chunk evidence bonus: +{bonus_total:.4f} across {len(distinct_bonus_chunks)} distinct chunk(s)."
                )
        else:
            diagnostics.append("Fallback retrieval scores used (cross-encoder not applied).")

        ranked_docs.append(
            RankedDocument(
                file_id=fid,
                filename=filename,
                doc_type=doc_type,
                group_id=resolved_gid,
                group_name=group_name,
                document_score=doc_score,
                rank=0,  # Assigned after sorting
                supporting_chunks=file_chunks,
                source_locations=locations_set,
                excerpts=excerpts,
                chunk_count=len(file_chunks),
                keyword_matches_count=kw_count,
                semantic_matches_count=sem_count,
                consensus_matches_count=consensus_count,
                is_reranked=is_reranked,
                diagnostics=diagnostics,
            )
        )

    # Sort documents descending by document_score, then chunk count, then file_id for determinism
    ranked_docs.sort(
        key=lambda d: (-d.document_score, -d.chunk_count, d.file_id)
    )

    # Assign 1-based ranks
    for idx, doc in enumerate(ranked_docs, start=1):
        doc.rank = idx

    if top_k_docs is not None and top_k_docs > 0:
        ranked_docs = ranked_docs[:top_k_docs]

    return ranked_docs


# ---------------------------------------------------------------------------
# End-to-End Search & Ranking Orchestration
# ---------------------------------------------------------------------------

def search_ranked_documents(
    query: str,
    limit: int = 10,
    group_id: int | None = None,
    rerank_top_k: int = 20,
    embed_model: str = DEFAULT_EMBED_MODEL,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    chroma_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    settings: Settings | None = None,
    custom_embedder: Callable[[list[str]], list[list[float]]] | None = None,
    custom_reranker: Callable[[list[tuple[str, str]]], list[float]] | None = None,
    disable_reranker: bool = False,
    whatsapp_filters: Any | None = None,
    context_weight: float | None = None,
    disable_context_ranking: bool = False,
) -> SearchRankedResult:
    """Orchestrate hybrid retrieval, cross-encoder reranking, and document-level ranking.

    Executes:
    1. Hybrid candidate search (FTS5 keyword + ChromaDB semantic).
    2. Cross-encoder scoring and candidate reranking (BAAI/bge-reranker-base).
    3. Document-level grouping and bounded relevance aggregation.
    4. WhatsApp conversational context retrieval for candidate documents.
    5. Context-aware filtering (groups, senders, dates, linked messages, query).
    6. Context-aware ranking (conservative WhatsApp context boost without overpowering content).
    7. Seamless fallback to hybrid candidate ranking if reranker fails or is disabled.

    Args:
        query: User search query or syllabus topic.
        limit: Maximum number of ranked documents to return.
        group_id: Optional study group constraint.
        rerank_top_k: Maximum candidate chunks to retrieve and feed into reranker.
        embed_model: Embedding model name.
        rerank_model: Reranker model identifier.
        chroma_dir: Optional custom ChromaDB directory.
        db_path: Optional custom SQLite database path.
        settings: Optional custom Settings.
        custom_embedder: Optional custom embedding function for testing.
        custom_reranker: Optional custom scoring function for testing.
        disable_reranker: If True, bypass cross-encoder and rank directly on hybrid signals.
        whatsapp_filters: Optional WhatsAppFilterCriteria to filter documents by WhatsApp context.
        context_weight: Optional custom weight for WhatsApp context scoring (default: 0.08).
        disable_context_ranking: If True, bypass WhatsApp context ranking adjustments.

    Returns:
        SearchRankedResult instance with ranked documents, supporting chunks, context, and diagnostics.
    """
    clean_query = query.strip() if query else ""
    if not clean_query:
        return SearchRankedResult(
            query=query,
            documents=[],
            chunks=[],
            total_candidates=0,
            is_reranked=False,
            retrieval_mode="empty",
            diagnostics=["Search query was empty."],
            applied_filters=whatsapp_filters.to_dict() if whatsapp_filters and hasattr(whatsapp_filters, "to_dict") else None,
        )

    active_settings = settings or load()
    effective_disable = disable_reranker or getattr(active_settings, "disable_reranker", False)
    effective_device = getattr(active_settings, "rerank_device", "")
    effective_rerank_model = rerank_model or getattr(active_settings, "rerank_model", DEFAULT_RERANK_MODEL)
    effective_rerank_top_k = max(rerank_top_k, limit * 2)

    # 1. Hybrid Candidate Retrieval
    hybrid_candidates: list[HybridSearchResult] = []
    try:
        hybrid_candidates = search_chunks_hybrid(
            query=clean_query,
            limit=effective_rerank_top_k,
            group_id=group_id,
            embed_model=embed_model,
            chroma_dir=chroma_dir,
            db_path=db_path,
            settings=active_settings,
            custom_embedder=custom_embedder,
        )
    except Exception as err:
        logger.error("Hybrid search retrieval failed for query '%s': %s", clean_query, err)
        return SearchRankedResult(
            query=clean_query,
            documents=[],
            chunks=[],
            total_candidates=0,
            is_reranked=False,
            retrieval_mode="error",
            diagnostics=[f"Hybrid retrieval error: {err}"],
            applied_filters=whatsapp_filters.to_dict() if whatsapp_filters and hasattr(whatsapp_filters, "to_dict") else None,
        )

    if not hybrid_candidates:
        return SearchRankedResult(
            query=clean_query,
            documents=[],
            chunks=[],
            total_candidates=0,
            is_reranked=False,
            retrieval_mode="empty",
            diagnostics=["No matching candidate chunks found across keyword and semantic indices."],
            applied_filters=whatsapp_filters.to_dict() if whatsapp_filters and hasattr(whatsapp_filters, "to_dict") else None,
        )

    # 2. Cross-Encoder Reranking
    processed_chunks: list[Any] = []
    is_reranked = False
    retrieval_mode = "hybrid_reranked"
    diagnostics: list[str] = [
        f"Hybrid retrieval retrieved {len(hybrid_candidates)} candidate chunk(s)."
    ]

    if effective_disable:
        diagnostics.append("Cross-encoder reranker bypassed by configuration (hybrid fallback active).")
        processed_chunks = hybrid_candidates
        retrieval_mode = "hybrid_fallback"
    else:
        try:
            processed_chunks = rerank_chunks(
                query=clean_query,
                candidates=hybrid_candidates,
                top_k=effective_rerank_top_k,
                model_name=effective_rerank_model,
                device=effective_device,
                custom_scorer=custom_reranker,
            )
            is_reranked = True
            diagnostics.append(
                f"Successfully reranked {len(processed_chunks)} chunk(s) with '{effective_rerank_model}'."
            )
        except Exception as err:
            logger.warning(
                "Cross-encoder reranking failed; falling back to hybrid candidate scores: %s",
                err,
            )
            diagnostics.append(
                f"Reranker unavailable: {err}. Falling back to hybrid retrieval scores."
            )
            processed_chunks = hybrid_candidates
            retrieval_mode = "hybrid_fallback"

    # 3. Document-Level Aggregation
    has_active_filters = whatsapp_filters is not None and getattr(whatsapp_filters, "is_active", lambda: False)()
    effective_disable_context = disable_context_ranking or getattr(active_settings, "disable_context_ranking", False)
    agg_limit = None if (has_active_filters or not effective_disable_context) else limit

    ranked_docs = rank_documents_from_chunks(
        chunks=processed_chunks,
        group_id=group_id,
        db_path=db_path,
        top_k_docs=agg_limit,
    )

    # 4. WhatsApp Context Retrieval
    context_map: dict[int, Any] = {}
    if ranked_docs:
        try:
            from core.whatsapp_context import get_whatsapp_context_for_documents
            file_ids = [doc.file_id for doc in ranked_docs]
            context_map = get_whatsapp_context_for_documents(file_ids, db_path=db_path)
            for doc in ranked_docs:
                doc.whatsapp_context = context_map.get(doc.file_id)
            diagnostics.append(
                f"Retrieved WhatsApp conversational context for {len(ranked_docs)} document(s)."
            )
        except Exception as err:
            logger.warning("Failed to retrieve WhatsApp context for search results: %s", err)
            diagnostics.append(f"WhatsApp context retrieval failed: {err}")

    # 5. Context-Aware Filtering
    excluded_count = 0
    if has_active_filters and ranked_docs:
        try:
            from core.whatsapp_context import filter_ranked_documents
            surviving_docs, excluded_records = filter_ranked_documents(
                documents=ranked_docs,
                criteria=whatsapp_filters,
                context_map=context_map,
                db_path=db_path,
            )
            excluded_count = len(excluded_records)
            diagnostics.append(
                f"Context filtering applied: {len(surviving_docs)} passed, {excluded_count} excluded."
            )
            for _, eval_res in excluded_records[:5]:
                diagnostics.append(f"Filter exclusion: {eval_res.reason}")

            # Re-assign deterministic sequential ranks to surviving documents
            for idx, doc in enumerate(surviving_docs, start=1):
                doc.rank = idx
            ranked_docs = surviving_docs
        except Exception as err:
            logger.warning("WhatsApp context filtering encountered an error: %s", err)
            diagnostics.append(f"Context filtering error: {err}. Preserved unfiltered documents.")

    # 6. Context-Aware Ranking (Step 4.2)
    is_context_ranked = False
    context_adj_count = 0
    if ranked_docs:
        try:
            from core.context_ranking import DEFAULT_CONTEXT_WEIGHT, apply_context_ranking
            effective_context_weight = (
                context_weight
                if context_weight is not None
                else getattr(active_settings, "context_ranking_weight", DEFAULT_CONTEXT_WEIGHT)
            )

            ranked_docs, adjustments = apply_context_ranking(
                documents=ranked_docs,
                query=clean_query,
                context_map=context_map,
                weight=effective_context_weight,
                disable=effective_disable_context,
            )
            if not effective_disable_context:
                is_context_ranked = True
                context_adj_count = sum(
                    1 for adj in adjustments if adj.original_rank != adj.final_rank or adj.boost > 0.0001
                )
                rank_changed_count = sum(1 for adj in adjustments if adj.original_rank != adj.final_rank)
                diagnostics.append(
                    f"Context-aware ranking applied (weight={effective_context_weight:.2f}): "
                    f"{rank_changed_count} rank reordering(s) across {len(ranked_docs)} document(s)."
                )
            else:
                diagnostics.append("Context-aware ranking bypassed by configuration.")
        except Exception as err:
            logger.warning("WhatsApp context ranking encountered an error: %s", err)
            diagnostics.append(f"Context ranking error: {err}. Preserved pure content ranking.")
            for doc in ranked_docs:
                if getattr(doc, "content_score", None) is None:
                    doc.content_score = doc.document_score
                if getattr(doc, "content_rank", None) is None:
                    doc.content_rank = doc.rank
                doc.adjustment_explanation = "Ranked by document content relevance alone (context ranking encountered an error)."

    # Truncate to user-requested limit after filtering and context-aware ranking
    ranked_docs = ranked_docs[:limit]

    applied_filters_dict = None
    if whatsapp_filters and hasattr(whatsapp_filters, "to_dict"):
        applied_filters_dict = whatsapp_filters.to_dict()

    return SearchRankedResult(
        query=clean_query,
        documents=ranked_docs,
        chunks=processed_chunks,
        total_candidates=len(hybrid_candidates),
        is_reranked=is_reranked,
        retrieval_mode=retrieval_mode,
        diagnostics=diagnostics,
        applied_filters=applied_filters_dict,
        excluded_documents_count=excluded_count,
        is_context_ranked=is_context_ranked,
        context_adjustments_count=context_adj_count,
    )
