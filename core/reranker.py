"""Local cross-encoder reranking module for StudyRAG.

Manages lazy loading of Hugging Face CrossEncoder models (specifically BAAI/bge-reranker-base),
query-passage pair generation, candidate deduplication, deterministic score assignment,
and preservation of multi-source retrieval signals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Any, Callable

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None  # type: ignore[assignment]

from core.config import Settings, load
from core.vector_store import HybridSearchResult

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"

__all__ = [
    "DEFAULT_RERANK_MODEL",
    "LocalRerankerManager",
    "RerankedChunk",
    "rerank_chunks",
]


# ---------------------------------------------------------------------------
# Data Transfer Object
# ---------------------------------------------------------------------------

@dataclass
class RerankedChunk:
    """A search candidate chunk scored and ranked by the local cross-encoder."""

    chunk_id: str
    file_id: int
    filename: str
    group_id: int
    text: str
    location: str
    page: int | None
    matched_by: str
    rerank_score: float
    rank: int
    keyword_score: float | None = None
    semantic_distance: float | None = None
    semantic_similarity: float | None = None
    sender: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass instance to a standard dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Local Reranker Manager (Singleton)
# ---------------------------------------------------------------------------

class LocalRerankerManager:
    """Singleton manager for local CrossEncoder models with lazy loading and device selection."""

    _instance: LocalRerankerManager | None = None
    _cached_model: Any = None
    _cached_model_name: str | None = None
    _cached_device: str | None = None

    @classmethod
    def get_instance(cls) -> LocalRerankerManager:
        """Return the shared singleton manager instance."""
        if cls._instance is None:
            cls._instance = LocalRerankerManager()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (primarily for testing and isolation)."""
        cls._instance = None
        cls._cached_model = None
        cls._cached_model_name = None
        cls._cached_device = None

    def resolve_device(self, requested_device: str = "") -> str:
        """Resolve target device (cuda, cpu) safely."""
        req = requested_device.strip().lower()
        if req:
            return req

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        return "cpu"

    def get_model(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        device: str = "",
    ) -> Any:
        """Retrieve or lazily initialize the SentenceTransformers CrossEncoder model.

        Args:
            model_name: Hugging Face model identifier (default: BAAI/bge-reranker-base).
            device: Optional explicit compute device ('cpu', 'cuda', or empty string for auto-detect).

        Returns:
            An initialized CrossEncoder model.

        Raises:
            ImportError: If sentence-transformers is not installed.
            RuntimeError: If model downloading, initialization, or weight loading fails.
        """
        if CrossEncoder is None:
            raise ImportError(
                "The 'sentence-transformers' package is not installed. "
                "Install it using 'pip install sentence-transformers'."
            )

        target_device = self.resolve_device(device)

        if (
            self._cached_model is not None
            and self._cached_model_name == model_name
            and self._cached_device == target_device
        ):
            return self._cached_model

        logger.info(
            "Loading local cross-encoder reranker model '%s' on device '%s'...",
            model_name,
            target_device,
        )

        try:
            model = CrossEncoder(model_name, device=target_device)
            self._cached_model = model
            self._cached_model_name = model_name
            self._cached_device = target_device
            return model
        except Exception as err:
            logger.error(
                "Failed to load cross-encoder model '%s' on device '%s': %s",
                model_name,
                target_device,
                err,
            )
            raise RuntimeError(
                f"Failed to load reranker model '{model_name}' on device '{target_device}': {err}"
            ) from err


# ---------------------------------------------------------------------------
# Cross-Encoder Reranking
# ---------------------------------------------------------------------------

def _extract_candidate_attrs(
    candidate: HybridSearchResult | dict[str, Any],
) -> dict[str, Any] | None:
    """Safely extract candidate attributes from either a dataclass or a dictionary."""
    if isinstance(candidate, HybridSearchResult):
        return {
            "chunk_id": candidate.chunk_id,
            "file_id": candidate.file_id,
            "filename": candidate.filename,
            "group_id": candidate.group_id,
            "text": candidate.text,
            "location": candidate.location,
            "page": candidate.page,
            "matched_by": candidate.matched_by,
            "keyword_score": candidate.keyword_score,
            "semantic_distance": candidate.semantic_distance,
            "semantic_similarity": candidate.semantic_similarity,
            "sender": candidate.sender,
        }
    if isinstance(candidate, dict):
        chunk_id = candidate.get("chunk_id") or candidate.get("id")
        text = candidate.get("text")
        if not chunk_id or text is None:
            return None
        return {
            "chunk_id": str(chunk_id),
            "file_id": int(candidate.get("file_id", 0)),
            "filename": str(candidate.get("filename", "")),
            "group_id": int(candidate.get("group_id", 0)),
            "text": str(text),
            "location": str(candidate.get("location", "")),
            "page": int(candidate["page"]) if candidate.get("page") is not None else None,
            "matched_by": str(candidate.get("matched_by", "unknown")),
            "keyword_score": float(candidate["keyword_score"]) if candidate.get("keyword_score") is not None else None,
            "semantic_distance": float(candidate["semantic_distance"]) if candidate.get("semantic_distance") is not None else None,
            "semantic_similarity": float(candidate["semantic_similarity"]) if candidate.get("semantic_similarity") is not None else None,
            "sender": str(candidate["sender"]) if candidate.get("sender") else None,
        }
    return None


def rerank_chunks(
    query: str,
    candidates: list[HybridSearchResult | dict[str, Any]],
    top_k: int | None = None,
    model_name: str = DEFAULT_RERANK_MODEL,
    device: str = "",
    custom_scorer: Callable[[list[tuple[str, str]]], list[float]] | None = None,
) -> list[RerankedChunk]:
    """Rerank hybrid search candidates using a local cross-encoder model.

    Computes query-chunk relevance scores directly across full token interactions,
    sorts candidates in descending order, assigns deterministic ranks, and preserves
    original retrieval metadata and signals.

    Args:
        query: User search query or syllabus topic.
        candidates: List of candidate chunks from hybrid retrieval (or dicts).
        top_k: Optional limit on the number of top reranked chunks to return.
        model_name: Hugging Face model identifier (default: BAAI/bge-reranker-base).
        device: Optional target device ('cpu', 'cuda', or empty string for auto-detect).
        custom_scorer: Optional custom scoring function for deterministic testing without loading model.

    Returns:
        List of RerankedChunk objects ordered by rerank_score descending.

    Raises:
        ValueError: If query is empty.
        RuntimeError: If model loading or inference fails.
    """
    clean_query = query.strip() if query else ""
    if not clean_query:
        raise ValueError("Query cannot be empty.")

    if not candidates:
        return []

    # 1. Validate and deduplicate candidate records
    seen_ids: set[str] = set()
    valid_candidates: list[dict[str, Any]] = []

    for raw_cand in candidates:
        attrs = _extract_candidate_attrs(raw_cand)
        if not attrs:
            logger.warning("Skipping malformed candidate without valid chunk_id or text: %s", raw_cand)
            continue

        cid = attrs["chunk_id"]
        if cid in seen_ids:
            logger.debug("Skipping duplicate candidate chunk_id: %s", cid)
            continue

        raw_text = attrs["text"].strip()
        if not raw_text:
            logger.warning("Skipping candidate with empty text (chunk_id: %s)", cid)
            continue

        attrs["text"] = raw_text
        seen_ids.add(cid)
        valid_candidates.append(attrs)

    if not valid_candidates:
        return []

    # 2. Build query-passage pairs
    pairs = [(clean_query, cand["text"]) for cand in valid_candidates]

    # 3. Predict cross-encoder relevance scores
    if custom_scorer is not None:
        try:
            scores = custom_scorer(pairs)
        except Exception as err:
            logger.error("Custom reranker scoring failed: %s", err)
            raise RuntimeError(f"Reranker scoring failed: {err}") from err
    else:
        model = LocalRerankerManager.get_instance().get_model(model_name=model_name, device=device)
        try:
            raw_scores = model.predict(pairs, show_progress_bar=False)
            if hasattr(raw_scores, "tolist"):
                scores = raw_scores.tolist()
            elif isinstance(raw_scores, (int, float)):
                scores = [float(raw_scores)]
            else:
                scores = [float(s) for s in raw_scores]
        except Exception as err:
            logger.error("Cross-encoder model inference failed: %s", err)
            raise RuntimeError(f"Cross-encoder inference failed on query '{clean_query}': {err}") from err

    if len(scores) != len(valid_candidates):
        raise RuntimeError(
            f"Cross-encoder returned {len(scores)} scores for {len(valid_candidates)} candidates."
        )

    # 4. Pair scores with candidate attributes
    scored_items: list[tuple[float, dict[str, Any]]] = [
        (float(score), cand) for score, cand in zip(scores, valid_candidates)
    ]

    # 5. Deterministic descending sort by rerank_score, using chunk_id as tie-breaker
    scored_items.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))

    # 6. Apply top_k truncation if specified
    if top_k is not None and top_k > 0:
        scored_items = scored_items[:top_k]

    # 7. Construct RerankedChunk instances with deterministic 1-based ranks
    results: list[RerankedChunk] = []
    for rank_idx, (score, cand) in enumerate(scored_items, start=1):
        results.append(
            RerankedChunk(
                chunk_id=cand["chunk_id"],
                file_id=cand["file_id"],
                filename=cand["filename"],
                group_id=cand["group_id"],
                text=cand["text"],
                location=cand["location"],
                page=cand["page"],
                matched_by=cand["matched_by"],
                rerank_score=score,
                rank=rank_idx,
                keyword_score=cand["keyword_score"],
                semantic_distance=cand["semantic_distance"],
                semantic_similarity=cand["semantic_similarity"],
                sender=cand["sender"],
            )
        )

    return results
