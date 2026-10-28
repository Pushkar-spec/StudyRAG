"""Tests for StudyRAG document-level ranking and end-to-end search orchestration (Phase 3, Step 3.2)."""

import os
from pathlib import Path
import pytest

from core.db import (
    db_session,
    get_or_create_group,
    init_db,
    insert_file,
    record_chunks_indexed_batch,
    update_file_indexing_status,
    update_file_processing_status,
)


def add_test_chunk(
    chunk_id: str,
    file_id: int,
    text: str,
    location: str = "",
    page: int | None = None,
    chunk_index: int = 0,
    db_path: Path | str | None = None,
) -> None:
    """Helper to insert a chunk directly into SQLite chunks table."""
    with db_session(db_path) as conn:
        conn.cursor().execute(
            """
            INSERT INTO chunks (id, file_id, page, text, location, chunk_index)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (chunk_id, file_id, page, text, location, chunk_index),
        )
from core.document_ranking import (
    RankedDocument,
    SearchRankedResult,
    _generate_excerpt,
    _is_near_duplicate,
    rank_documents_from_chunks,
    search_ranked_documents,
)
from core.reranker import RerankedChunk
from core.vector_store import (
    HybridSearchResult,
    get_or_create_collection,
)


@pytest.fixture
def ranking_db(tmp_path: Path):
    """Setup isolated SQLite database with registered files."""
    db_path = tmp_path / "ranking_test.db"
    init_db(db_path)

    gid1 = get_or_create_group("CS_101", db_path=db_path)
    gid2 = get_or_create_group("CS_102", db_path=db_path)

    f1 = insert_file(
        group_id=gid1,
        sha256="hash_doc1",
        filename="DBMS_Normalization.pdf",
        path=str(tmp_path / "DBMS_Normalization.pdf"),
        ext=".pdf",
        db_path=db_path,
    )
    f2 = insert_file(
        group_id=gid1,
        sha256="hash_doc2",
        filename="DBMS_Decomposition.docx",
        path=str(tmp_path / "DBMS_Decomposition.docx"),
        ext=".docx",
        db_path=db_path,
    )
    f3 = insert_file(
        group_id=gid2,
        sha256="hash_doc3",
        filename="Cryptography_Overview.pdf",
        path=str(tmp_path / "Cryptography_Overview.pdf"),
        ext=".pdf",
        db_path=db_path,
    )

    return {
        "db_path": db_path,
        "tmp_path": tmp_path,
        "gid1": gid1,
        "gid2": gid2,
        "f1": f1,
        "f2": f2,
        "f3": f3,
    }


# ---------------------------------------------------------------------------
# 1. Helper Unit Tests
# ---------------------------------------------------------------------------

def test_near_duplicate_detection():
    """Verify that sliding window overlapping chunks are identified as near duplicates."""
    t1 = "Database normalization aims to eliminate redundancy and update anomalies in relational schemas."
    t2 = "aims to eliminate redundancy and update anomalies in relational schemas using functional dependencies."
    # Significant overlap
    assert _is_near_duplicate(t1, t2, threshold=0.6) is True

    t3 = "Public key cryptography utilizes asymmetric key pairs including RSA and elliptic curve algorithms."
    assert _is_near_duplicate(t1, t3, threshold=0.5) is False
    assert _is_near_duplicate("", t1) is False


def test_excerpt_generation():
    """Excerpts are trimmed to single-line strings with max character caps."""
    multiline = "Line 1.\n\nLine 2 with extra    spaces.\nLine 3."
    excerpt = _generate_excerpt(multiline, max_chars=40)
    assert "\n" not in excerpt
    assert "Line 1. Line 2 with extra spaces." in excerpt or "..." in excerpt
    assert len(excerpt) <= 43  # max_chars + '...'


# ---------------------------------------------------------------------------
# 2. Document-Level Aggregation Tests
# ---------------------------------------------------------------------------

def test_rank_documents_groups_by_file_id_and_resolves_metadata(ranking_db: dict):
    """Candidate chunks belonging to different files are grouped and metadata resolved from SQLite."""
    f1 = ranking_db["f1"]
    f2 = ranking_db["f2"]
    db_path = ranking_db["db_path"]

    chunks = [
        RerankedChunk(
            chunk_id="chk_f1_0",
            file_id=f1,
            filename="",  # Intentionally blank to test SQLite metadata resolution
            group_id=0,
            text="First normal form requires atomic values in every column.",
            location="Page 1",
            page=1,
            matched_by="both",
            rerank_score=7.5,
            rank=1,
        ),
        RerankedChunk(
            chunk_id="chk_f1_1",
            file_id=f1,
            filename="",
            group_id=0,
            text="Second normal form eliminates partial key dependencies.",
            location="Page 2",
            page=2,
            matched_by="semantic",
            rerank_score=6.0,
            rank=2,
        ),
        RerankedChunk(
            chunk_id="chk_f2_0",
            file_id=f2,
            filename="",
            group_id=0,
            text="Decomposition into BCNF guarantees lossless joins.",
            location="Paragraph 4",
            page=None,
            matched_by="keyword",
            rerank_score=4.0,
            rank=3,
        ),
    ]

    docs = rank_documents_from_chunks(chunks, db_path=db_path)

    assert len(docs) == 2
    # Document 1 has chunks f1
    d1 = docs[0]
    assert d1.file_id == f1
    assert d1.filename == "DBMS_Normalization.pdf"
    assert d1.doc_type == ".pdf"
    assert d1.group_name == "CS_101"
    assert d1.rank == 1
    assert d1.chunk_count == 2
    assert d1.source_locations == ["Page 1", "Page 2"]
    assert d1.consensus_matches_count == 1
    assert d1.semantic_matches_count == 1

    # Document 2 has chunk f2
    d2 = docs[1]
    assert d2.file_id == f2
    assert d2.filename == "DBMS_Decomposition.docx"
    assert d2.doc_type == ".docx"
    assert d2.rank == 2
    assert d2.chunk_count == 1
    assert d2.source_locations == ["Paragraph 4"]


def test_bounded_score_aggregation_and_non_inflation():
    """Verify that a document cannot dominate simply by possessing many weak chunks."""
    # Doc A: 1 genuinely strong chunk (score = 8.0)
    chunk_a = RerankedChunk(
        chunk_id="chk_a_0",
        file_id=1,
        filename="Target_Paper.pdf",
        group_id=1,
        text="Comprehensive theorem proof of third normal form synthesis.",
        location="Page 5",
        page=5,
        matched_by="both",
        rerank_score=8.0,
        rank=1,
    )

    # Doc B: 50 weak chunks (score = 1.0 each) across distinct topics
    distinct_topics = [
        "Relational algebra covers selection, projection, and cartesian product operations.",
        "SQL queries employ group by and having clauses for aggregating summary tables.",
        "B-tree indexing organizes disk blocks into balanced hierarchical search trees.",
        "Transaction ACID properties ensure atomicity, consistency, isolation, and durability.",
    ]
    chunks_b = [
        RerankedChunk(
            chunk_id=f"chk_b_{i}",
            file_id=2,
            filename="Large_Textbook.pdf",
            group_id=1,
            text=distinct_topics[i % len(distinct_topics)] + f" (section {i})",
            location=f"Page {i+1}",
            page=i+1,
            matched_by="keyword",
            rerank_score=1.0,
            rank=i+2,
        )
        for i in range(50)
    ]

    all_chunks = [chunk_a] + chunks_b
    docs = rank_documents_from_chunks(all_chunks)

    # Doc A must rank first despite having only 1 chunk vs 50 chunks!
    assert docs[0].file_id == 1
    assert docs[0].rank == 1
    assert docs[0].document_score == 8.0

    assert docs[1].file_id == 2
    assert docs[1].rank == 2
    # Doc B score: s1 (1.0) + w2*1.0 (0.10) + w3*1.0 (0.05) + w4*1.0 (0.02) = 1.17
    assert 1.15 <= docs[1].document_score <= 1.20


def test_near_duplicate_chunks_do_not_inflate_score():
    """Overlapping sliding-window chunks from the same page do not add bonus."""
    c1 = RerankedChunk(
        chunk_id="c1",
        file_id=1,
        filename="doc.pdf",
        group_id=1,
        text="Relational database normalization is the process of structuring relational schema.",
        location="Page 1",
        page=1,
        matched_by="both",
        rerank_score=5.0,
        rank=1,
    )
    # Near duplicate of c1 (>70% overlap)
    c2 = RerankedChunk(
        chunk_id="c2",
        file_id=1,
        filename="doc.pdf",
        group_id=1,
        text="database normalization is the process of structuring relational schema to reduce redundancy.",
        location="Page 1",
        page=1,
        matched_by="both",
        rerank_score=4.9,
        rank=2,
    )

    docs = rank_documents_from_chunks([c1, c2])
    assert len(docs) == 1
    # Because c2 is a near-duplicate, bonus is skipped and score remains 5.0
    assert docs[0].document_score == 5.0


def test_distinct_evidence_provides_bounded_bonus():
    """Truly distinct chunks from different pages contribute a bounded score bonus."""
    c1 = RerankedChunk(
        chunk_id="c1",
        file_id=1,
        filename="doc.pdf",
        group_id=1,
        text="Unit 1 covers First Normal Form and table flat structure.",
        location="Page 1",
        page=1,
        matched_by="both",
        rerank_score=5.0,
        rank=1,
    )
    c2 = RerankedChunk(
        chunk_id="c2",
        file_id=1,
        filename="doc.pdf",
        group_id=1,
        text="Unit 4 analyzes Boyce-Codd Normal Form anomalies and decomposition algorithms.",
        location="Page 4",
        page=4,
        matched_by="both",
        rerank_score=4.0,
        rank=2,
    )

    docs = rank_documents_from_chunks([c1, c2])
    assert len(docs) == 1
    # Expected score: 5.0 + 0.10 * 4.0 = 5.40
    assert docs[0].document_score == pytest.approx(5.40, rel=1e-3)


# ---------------------------------------------------------------------------
# 3. End-to-End Orchestrator Integration Tests
# ---------------------------------------------------------------------------

def test_search_ranked_documents_end_to_end_with_mocked_reranker(ranking_db: dict):
    """Verify complete pipeline: hybrid search -> reranker -> document ranking."""
    db_path = ranking_db["db_path"]
    f1 = ranking_db["f1"]
    f2 = ranking_db["f2"]

    # Insert searchable chunks into SQLite
    add_test_chunk(
        chunk_id="chk_dbms_1",
        file_id=f1,
        text="DBMS normalization decomposes relations to eliminate functional dependency anomalies.",
        location="Page 1",
        page=1,
        chunk_index=0,
        db_path=db_path,
    )
    add_test_chunk(
        chunk_id="chk_dbms_2",
        file_id=f2,
        text="Decomposition into third normal form avoids insertion and deletion anomalies.",
        location="Page 2",
        page=2,
        chunk_index=0,
        db_path=db_path,
    )
    update_file_processing_status(f1, "processed", chunk_count=1, db_path=db_path)
    update_file_processing_status(f2, "processed", chunk_count=1, db_path=db_path)

    # Scorer prioritizing f1
    def mock_reranker(pairs):
        scores = []
        for q, text in pairs:
            if "functional dependency" in text:
                scores.append(9.5)
            else:
                scores.append(4.2)
        return scores

    res = search_ranked_documents(
        query="normalization functional dependency",
        db_path=db_path,
        custom_reranker=mock_reranker,
    )

    assert isinstance(res, SearchRankedResult)
    assert res.is_reranked is True
    assert res.retrieval_mode == "hybrid_reranked"
    assert len(res.documents) >= 1
    # Leading document must be f1
    assert res.documents[0].file_id == f1
    assert res.documents[0].filename == "DBMS_Normalization.pdf"
    assert res.documents[0].rank == 1


def test_search_ranked_documents_fallback_when_reranker_fails(ranking_db: dict):
    """When reranking fails, hybrid candidates are preserved and used for fallback ranking."""
    db_path = ranking_db["db_path"]
    f1 = ranking_db["f1"]

    add_test_chunk(
        chunk_id="chk_fallback_1",
        file_id=f1,
        text="Relational algebra query optimization and index scans.",
        location="Page 1",
        page=1,
        chunk_index=0,
        db_path=db_path,
    )
    update_file_processing_status(f1, "processed", chunk_count=1, db_path=db_path)

    def failing_reranker(pairs):
        raise RuntimeError("Simulated GPU OOM / model loading error")

    res = search_ranked_documents(
        query="relational algebra",
        db_path=db_path,
        custom_reranker=failing_reranker,
    )

    # Pipeline must NOT crash
    assert res.is_reranked is False
    assert res.retrieval_mode == "hybrid_fallback"
    assert len(res.documents) == 1
    assert res.documents[0].file_id == f1
    assert any("Reranker unavailable" in d for d in res.diagnostics)


def test_search_ranked_documents_empty_query():
    """Empty query returns empty SearchRankedResult immediately."""
    res = search_ranked_documents("   ")
    assert res.documents == []
    assert res.retrieval_mode == "empty"


# ---------------------------------------------------------------------------
# 4. Section 8 End-to-End Acceptance Scenario
# ---------------------------------------------------------------------------

def test_section_8_end_to_end_acceptance_scenario(tmp_path: Path):
    """Full execution of the Section 8 acceptance scenario:

    1. Process 3 synthetic documents:
       - Doc 1 (DBMS): normalization, functional dependencies, 1NF, 2NF, 3NF.
       - Doc 2 (DBMS): decomposition, redundancy, update anomalies.
       - Doc 3 (Crypto): RSA encryption, public keys, ciphertexts.
    2. Query: 'DBMS normalization'
    3. Verify hybrid search returns candidate chunks.
    4. Reranker scores actual query-chunk pairs.
    5. Document aggregation produces 1 result per original file.
    6. Leading results include traceable source locations and excerpts.
    7. Multiple chunks from same file aggregated correctly with bounded bonus.
    8. Unrelated crypto document ranked lowest or excluded.
    """
    db_path = tmp_path / "acceptance.db"
    init_db(db_path)

    gid = get_or_create_group("DBMS_Class", db_path=db_path)

    f1 = insert_file(
        group_id=gid,
        sha256="hash_synth_1",
        filename="Unit3_Normalization_Fundamentals.pdf",
        path=str(tmp_path / "doc1.pdf"),
        ext=".pdf",
        db_path=db_path,
    )
    f2 = insert_file(
        group_id=gid,
        sha256="hash_synth_2",
        filename="Unit4_Decomposition_and_Anomalies.docx",
        path=str(tmp_path / "doc2.docx"),
        ext=".docx",
        db_path=db_path,
    )
    f3 = insert_file(
        group_id=gid,
        sha256="hash_synth_3",
        filename="Security_RSA_Cryptography.pdf",
        path=str(tmp_path / "doc3.pdf"),
        ext=".pdf",
        db_path=db_path,
    )

    # Insert chunks
    add_test_chunk(
        chunk_id="chk_u3_1",
        file_id=f1,
        text="DBMS normalization is the formal technique for organizing relational database tables to prevent anomalies. 1NF, 2NF, and 3NF rely on functional dependencies.",
        location="Page 3",
        page=3,
        chunk_index=0,
        db_path=db_path,
    )
    add_test_chunk(
        chunk_id="chk_u3_2",
        file_id=f1,
        text="In DBMS normalization, Third Normal Form (3NF) states that no non-prime attribute may be transitively dependent on any candidate key.",
        location="Page 5",
        page=5,
        chunk_index=1,
        db_path=db_path,
    )
    add_test_chunk(
        chunk_id="chk_u4_1",
        file_id=f2,
        text="In DBMS relational schema design, lossless-join decomposition eliminates data redundancy and update anomalies during normalization.",
        location="Paragraph 12",
        page=None,
        chunk_index=0,
        db_path=db_path,
    )
    add_test_chunk(
        chunk_id="chk_crypto_1",
        file_id=f3,
        text="RSA public key encryption uses modular arithmetic and prime factorization for asymmetric ciphers.",
        location="Page 1",
        page=1,
        chunk_index=0,
        db_path=db_path,
    )

    for fid in (f1, f2, f3):
        update_file_processing_status(fid, "processed", chunk_count=1, db_path=db_path)

    # Deterministic mock reranker that rewards normalization concepts
    def mock_bge_reranker(pairs):
        scores = []
        for q, text in pairs:
            score = 0.0
            if "normalization" in text.lower():
                score += 5.0
            if "3nf" in text.lower():
                score += 3.5
            if "functional dependency" in text.lower():
                score += 2.0
            if "redundancy" in text.lower():
                score += 1.5
            if "rsa" in text.lower():
                score -= 4.0
            scores.append(score)
        return scores

    res = search_ranked_documents(
        query="DBMS normalization",
        db_path=db_path,
        custom_reranker=mock_bge_reranker,
    )

    assert res.is_reranked is True
    assert len(res.documents) >= 2

    # Leading document must be Unit3_Normalization_Fundamentals.pdf
    top_doc = res.documents[0]
    assert top_doc.file_id == f1
    assert top_doc.filename == "Unit3_Normalization_Fundamentals.pdf"
    assert top_doc.rank == 1
    assert "Page 3" in top_doc.source_locations
    assert len(top_doc.excerpts) >= 1

    # Second document should be Unit4_Decomposition_and_Anomalies.docx
    second_doc = res.documents[1]
    assert second_doc.file_id == f2
    assert second_doc.rank == 2

    # Verify score aggregation on top doc: s1 + bounded s2
    # chk_u3_1 (5.0+3.5=8.5), chk_u3_2 (5.0+3.5=8.5) -> s1 = 8.5, s2 = 8.5
    # Expected doc score = 8.5 + 0.10 * 8.5 = 9.35
    assert top_doc.document_score == pytest.approx(9.35, rel=1e-3)


# ---------------------------------------------------------------------------
# 5. Optional Real-Model Smoke Test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("RUN_REAL_RERANKER_SMOKE_TEST"),
    reason="Set RUN_REAL_RERANKER_SMOKE_TEST=1 to execute the real BAAI/bge-reranker-base smoke test.",
)
def test_real_model_smoke_test():
    """Optional smoke test loading the real BAAI/bge-reranker-base model.

    Verifies genuine Hugging Face weight loading and local inference on synthetic pairs.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
    pairs = [
        ("DBMS normalization", "3NF eliminates transitive dependencies in relational databases."),
        ("DBMS normalization", "RSA cryptography relies on prime factorization."),
    ]
    scores = model.predict(pairs)
    # The relevant passage must receive a higher score than the cryptography passage
    assert scores[0] > scores[1]
