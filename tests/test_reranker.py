"""Tests for StudyRAG cross-encoder reranking (Phase 3, Step 3.2)."""

from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from core.reranker import (
    DEFAULT_RERANK_MODEL,
    LocalRerankerManager,
    RerankedChunk,
    rerank_chunks,
)
from core.vector_store import HybridSearchResult


def test_reranker_manager_singleton_and_device_resolution():
    """Verify singleton pattern, reset behavior, and safe device resolution."""
    LocalRerankerManager.reset_instance()
    mgr1 = LocalRerankerManager.get_instance()
    mgr2 = LocalRerankerManager.get_instance()
    assert mgr1 is mgr2

    # Explicit device resolution
    assert mgr1.resolve_device("cpu") == "cpu"
    assert mgr1.resolve_device("cuda") == "cuda"
    assert mgr1.resolve_device("") in ("cpu", "cuda")

    LocalRerankerManager.reset_instance()
    assert LocalRerankerManager._instance is None


def test_rerank_chunks_validation_empty_query():
    """Empty or whitespace-only queries must raise ValueError."""
    with pytest.raises(ValueError, match="Query cannot be empty"):
        rerank_chunks("", [])

    with pytest.raises(ValueError, match="Query cannot be empty"):
        rerank_chunks("   ", [{"chunk_id": "c1", "text": "hello"}])


def test_rerank_chunks_empty_candidates():
    """Empty candidate list safely returns an empty list without loading model."""
    res = rerank_chunks("normalization", [])
    assert res == []


def test_rerank_chunks_query_passage_pair_construction_and_score_association():
    """Query-passage pairs are built correctly and scores are associated with matching candidates."""
    recorded_pairs = []

    def mock_scorer(pairs):
        nonlocal recorded_pairs
        recorded_pairs = pairs
        # Return deterministically distinct scores
        return [2.5, 8.1, -1.2]

    candidates = [
        HybridSearchResult(
            chunk_id="chk_1",
            file_id=1,
            filename="notes.pdf",
            group_id=10,
            text="First passage about BCNF.",
            location="Page 1",
            page=1,
            matched_by="semantic",
            semantic_distance=0.2,
            semantic_similarity=0.8,
        ),
        HybridSearchResult(
            chunk_id="chk_2",
            file_id=1,
            filename="notes.pdf",
            group_id=10,
            text="Second passage discussing 3NF and Boyce-Codd.",
            location="Page 2",
            page=2,
            matched_by="both",
            keyword_score=0.9,
            semantic_distance=0.1,
            semantic_similarity=0.9,
        ),
        HybridSearchResult(
            chunk_id="chk_3",
            file_id=2,
            filename="irrelevant.pdf",
            group_id=10,
            text="Third passage about network routing.",
            location="Page 1",
            page=1,
            matched_by="keyword",
            keyword_score=0.3,
        ),
    ]

    results = rerank_chunks(
        query="database normalization 3NF",
        candidates=candidates,
        custom_scorer=mock_scorer,
    )

    # Verify query-passage pairs passed to scorer
    assert len(recorded_pairs) == 3
    assert recorded_pairs[0] == ("database normalization 3NF", "First passage about BCNF.")
    assert recorded_pairs[1] == ("database normalization 3NF", "Second passage discussing 3NF and Boyce-Codd.")
    assert recorded_pairs[2] == ("database normalization 3NF", "Third passage about network routing.")

    # Result ordering must be sorted descending by rerank_score
    # chk_2 (8.1) > chk_1 (2.5) > chk_3 (-1.2)
    assert len(results) == 3
    assert results[0].chunk_id == "chk_2"
    assert results[0].rerank_score == 8.1
    assert results[0].rank == 1

    assert results[1].chunk_id == "chk_1"
    assert results[1].rerank_score == 2.5
    assert results[1].rank == 2

    assert results[2].chunk_id == "chk_3"
    assert results[2].rerank_score == -1.2
    assert results[2].rank == 3


def test_rerank_chunks_preserves_original_metadata_and_scores():
    """All retrieval metadata and original search scores must be preserved intact."""
    candidate = HybridSearchResult(
        chunk_id="chk_test_01",
        file_id=42,
        filename="lecture_db.pdf",
        group_id=7,
        text="Relational database functional dependencies.",
        location="Slide 5",
        page=5,
        matched_by="both",
        keyword_score=0.85,
        semantic_distance=0.15,
        semantic_similarity=0.85,
        sender="Prof. Smith",
    )

    results = rerank_chunks(
        query="functional dependency",
        candidates=[candidate],
        custom_scorer=lambda pairs: [4.2],
    )

    assert len(results) == 1
    r = results[0]
    assert r.chunk_id == "chk_test_01"
    assert r.file_id == 42
    assert r.filename == "lecture_db.pdf"
    assert r.group_id == 7
    assert r.text == "Relational database functional dependencies."
    assert r.location == "Slide 5"
    assert r.page == 5
    assert r.matched_by == "both"
    assert r.keyword_score == 0.85
    assert r.semantic_distance == 0.15
    assert r.semantic_similarity == 0.85
    assert r.sender == "Prof. Smith"
    assert r.rerank_score == 4.2
    assert r.rank == 1


def test_rerank_chunks_deduplicates_duplicate_candidate_ids():
    """Candidates sharing the same chunk_id are deduplicated before scoring."""
    candidates = [
        {"chunk_id": "c1", "text": "Unique text 1", "file_id": 1},
        {"chunk_id": "c1", "text": "Duplicate text 1", "file_id": 1},
        {"chunk_id": "c2", "text": "Unique text 2", "file_id": 1},
    ]

    scored_count = 0

    def mock_scorer(pairs):
        nonlocal scored_count
        scored_count = len(pairs)
        return [1.0, 2.0]

    results = rerank_chunks("query", candidates, custom_scorer=mock_scorer)
    assert scored_count == 2
    assert len(results) == 2
    assert {r.chunk_id for r in results} == {"c1", "c2"}


def test_rerank_chunks_skips_malformed_and_empty_text_candidates():
    """Candidates missing chunk_id or text, or with empty text, are safely omitted."""
    candidates = [
        {"chunk_id": "valid_1", "text": "Good text", "file_id": 1},
        {"text": "Missing chunk_id", "file_id": 1},  # Malformed
        {"chunk_id": "empty_2", "text": "   ", "file_id": 1},  # Empty text
        {"chunk_id": "valid_2", "text": "Another good passage", "file_id": 1},
    ]

    results = rerank_chunks("query", candidates, custom_scorer=lambda pairs: [1.0, 2.0])
    assert len(results) == 2
    assert [r.chunk_id for r in results] == ["valid_2", "valid_1"]


def test_rerank_chunks_top_k_truncation():
    """top_k limit strictly truncates the resulting ranked list."""
    candidates = [
        {"chunk_id": f"c_{i}", "text": f"Passage {i}", "file_id": 1}
        for i in range(10)
    ]

    # Return increasing scores so c_9 is highest
    scores = [float(i) for i in range(10)]
    results = rerank_chunks("query", candidates, top_k=3, custom_scorer=lambda pairs: scores)

    assert len(results) == 3
    assert [r.chunk_id for r in results] == ["c_9", "c_8", "c_7"]
    assert [r.rank for r in results] == [1, 2, 3]


def test_rerank_chunks_model_failure_raises_runtime_error():
    """Inference failure in cross-encoder raises descriptive RuntimeError."""
    def failing_scorer(pairs):
        raise ValueError("CUDA out of memory in tensor operation")

    with pytest.raises(RuntimeError, match="Reranker scoring failed"):
        rerank_chunks("query", [{"chunk_id": "c1", "text": "text"}], custom_scorer=failing_scorer)


def test_no_external_network_calls_during_reranking():
    """Standard automated tests must never perform network requests."""
    with patch("urllib.request.urlopen") as mock_url:
        mock_url.side_effect = AssertionError("Unexpected network call!")
        results = rerank_chunks(
            query="test",
            candidates=[{"chunk_id": "c1", "text": "local text"}],
            custom_scorer=lambda pairs: [3.5],
        )
        assert len(results) == 1
        assert results[0].rerank_score == 3.5
