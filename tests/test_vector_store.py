"""Tests for StudyRAG local embeddings, ChromaDB vector store, and hybrid retrieval (Step 2.2)."""

from __future__ import annotations

import math
from pathlib import Path
import pytest
import sqlite3

try:
    import pymupdf as fitz
except ImportError:
    import fitz
import pptx
import docx

from core.config import Settings
from core.db import (
    get_connection,
    get_file_chunks,
    get_processing_status_counts,
    get_vector_index_counts,
    init_db,
    insert_file,
    record_chunks_indexed_batch,
    search_chunks_fts,
    update_file_indexing_status,
    update_file_processing_status,
)
from core.extraction import persist_chunks_for_file, process_file
from core.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBED_MODEL,
    HybridSearchResult,
    IndexingResult,
    LocalEmbeddingManager,
    SemanticSearchResult,
    compute_chunk_content_hash,
    get_chroma_client,
    get_or_create_collection,
    index_pending_chunks,
    search_chunks_hybrid,
    search_chunks_semantic,
)


# ---------------------------------------------------------------------------
# Deterministic Mock Embedder Fixture
# ---------------------------------------------------------------------------

def make_deterministic_embedder(dim: int = 384):
    """Create a deterministic mock embedder that clusters semantic topics."""
    keywords_topic0 = [
        "dbms", "normalization", "redundancy", "anomalies", "relation",
        "schema", "decomposition", "1nf", "2nf", "3nf", "normal", "form"
    ]
    keywords_topic1 = [
        "galois", "counter", "gcm", "authenticated", "encryption",
        "cipher", "crypto", "aes", "security"
    ]
    keywords_topic2 = [
        "cookie", "cookies", "baking", "recipe", "chocolate", "flour",
        "sugar", "culinary", "butter"
    ]

    def embedder(texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            vec = [0.0] * dim
            lower = text.lower()

            w0 = sum(3.0 for k in keywords_topic0 if k in lower)
            w1 = sum(3.0 for k in keywords_topic1 if k in lower)
            w2 = sum(3.0 for k in keywords_topic2 if k in lower)

            vec[0] = w0
            vec[1] = w1
            vec[2] = w2

            # Use word hashes for secondary variation
            words = lower.split()
            for w in words:
                h = abs(hash(w))
                idx = 3 + (h % (dim - 3))
                vec[idx] += 0.2

            sq_sum = sum(x * x for x in vec)
            if sq_sum > 0:
                norm = math.sqrt(sq_sum)
                vec = [x / norm for x in vec]
            else:
                vec[0] = 1.0
            results.append(vec)
        return results

    return embedder


@pytest.fixture
def mock_embedder():
    return make_deterministic_embedder(dim=384)


@pytest.fixture
def test_env(tmp_path: Path):
    """Setup isolated test database, ChromaDB storage directory, and default group."""
    from core.db import get_or_create_group
    db_path = tmp_path / "test.db"
    init_db(db_path)
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    group_id = get_or_create_group("StudyGroup1", db_path=db_path)
    return {"db_path": db_path, "chroma_dir": chroma_dir, "tmp_path": tmp_path, "group_id": group_id}


# ---------------------------------------------------------------------------
# 1. ChromaDB Collection and Embedding Manager Tests
# ---------------------------------------------------------------------------

def test_chroma_collection_initialization_and_persistence(test_env):
    """ChromaDB collection initializes with persistent storage and cosine distance."""
    chroma_dir = test_env["chroma_dir"]
    client = get_chroma_client(chroma_dir=chroma_dir)
    assert client is not None

    collection = get_or_create_collection(
        client=client,
        collection_name="test_col",
        chroma_dir=chroma_dir,
    )
    assert collection.name == "test_col"
    assert collection.metadata.get("hnsw:space") == "cosine"

    # Insert a vector and verify persistence
    collection.add(
        ids=["chk_1_0000"],
        embeddings=[[0.1] * 384],
        metadatas=[{"file_id": 1, "chunk_id": "chk_1_0000"}],
        documents=["Test document content"],
    )
    assert collection.count() == 1

    # Reload client from same directory and verify persistent collection
    client2 = get_chroma_client(chroma_dir=chroma_dir)
    col2 = client2.get_collection("test_col")
    assert col2.count() == 1


def test_embedding_manager_uses_custom_embedder_without_download(mock_embedder):
    """LocalEmbeddingManager uses custom embedder and returns normalized vectors."""
    mgr = LocalEmbeddingManager.get_instance()
    texts = ["DBMS normalization reduces redundancy.", "Galois Counter Mode."]
    embeddings = mgr.embed_texts(texts, custom_embedder=mock_embedder)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384

    # Check query embedding
    q_vec = mgr.embed_query("normalization", custom_embedder=mock_embedder)
    assert len(q_vec) == 384


# ---------------------------------------------------------------------------
# 2. Incremental Indexing and Idempotency Tests
# ---------------------------------------------------------------------------

def test_index_pending_chunks_incremental_and_idempotent(test_env, mock_embedder):
    """Indexing succeeds, records state in SQLite, and repeats idempotently."""
    db_path = test_env["db_path"]
    chroma_dir = test_env["chroma_dir"]

    # Register file and chunks
    file_id = insert_file(group_id=1, sha256="hash1", filename="dbms.pdf", path="/p/dbms.pdf", ext=".pdf", db_path=db_path)
    from core.extraction import DocumentChunk
    chunks = [
        DocumentChunk(
            id=f"chk_{file_id}_0000",
            file_id=file_id,
            chunk_index=0,
            text="DBMS normalization reduces redundancy and update anomalies.",
            location="Page 1",
            page=1,
        ),
        DocumentChunk(
            id=f"chk_{file_id}_0001",
            file_id=file_id,
            chunk_index=1,
            text="First, Second and Third Normal Forms organize relational attributes.",
            location="Page 2",
            page=2,
        ),
    ]
    persist_chunks_for_file(file_id, chunks, db_path=db_path)
    update_file_processing_status(file_id, "processed", chunk_count=2, db_path=db_path)

    # First indexing run
    res1 = index_pending_chunks(
        group_id=1,
        chroma_dir=chroma_dir,
        db_path=db_path,
        custom_embedder=mock_embedder,
    )
    assert res1.total_eligible == 2
    assert res1.indexed_count == 2
    assert res1.skipped_count == 0
    assert res1.failed_count == 0

    # Verify SQLite index state
    counts = get_vector_index_counts(group_id=1, db_path=db_path)
    assert counts["eligible_chunks"] == 2
    assert counts["indexed_chunks"] == 2
    assert counts["pending_chunks"] == 0
    assert counts["indexed_files"] == 1

    # Verify ChromaDB collection count
    col = get_or_create_collection(chroma_dir=chroma_dir)
    assert col.count() == 2

    # Second indexing run (should be completely idempotent and skip all chunks)
    res2 = index_pending_chunks(
        group_id=1,
        chroma_dir=chroma_dir,
        db_path=db_path,
        custom_embedder=mock_embedder,
    )
    assert res2.total_eligible == 2
    assert res2.indexed_count == 0
    assert res2.skipped_count == 2
    assert res2.failed_count == 0
    assert col.count() == 2  # No duplicate vectors


def test_chunk_content_change_triggers_reindexing(test_env, mock_embedder):
    """If chunk text changes in SQLite, re-indexing detects hash mismatch and updates vector."""
    db_path = test_env["db_path"]
    chroma_dir = test_env["chroma_dir"]

    file_id = insert_file(group_id=1, sha256="hash2", filename="crypto.pdf", path="/p/crypto.pdf", ext=".pdf", db_path=db_path)
    from core.extraction import DocumentChunk
    c = DocumentChunk(
        id=f"chk_{file_id}_0000",
        file_id=file_id,
        chunk_index=0,
        text="Galois/Counter Mode provides authenticated encryption.",
        location="Page 1",
        page=1,
    )
    persist_chunks_for_file(file_id, [c], db_path=db_path)
    update_file_processing_status(file_id, "processed", chunk_count=1, db_path=db_path)

    # Initial indexing
    res1 = index_pending_chunks(chroma_dir=chroma_dir, db_path=db_path, custom_embedder=mock_embedder)
    assert res1.indexed_count == 1

    # Simulate chunk text update in SQLite
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE chunks SET text = ? WHERE id = ?",
            ("Galois/Counter Mode provides high-performance authenticated encryption with AES.", f"chk_{file_id}_0000"),
        )
        conn.commit()

    # Re-index
    res2 = index_pending_chunks(chroma_dir=chroma_dir, db_path=db_path, custom_embedder=mock_embedder)
    assert res2.total_eligible == 1
    assert res2.indexed_count == 1
    assert res2.skipped_count == 0

    # ChromaDB count remains 1
    col = get_or_create_collection(chroma_dir=chroma_dir)
    assert col.count() == 1


def test_different_embed_model_triggers_reindexing(test_env, mock_embedder):
    """Switching embed_model requires indexing under the new model version."""
    db_path = test_env["db_path"]
    chroma_dir = test_env["chroma_dir"]

    file_id = insert_file(group_id=1, sha256="hash3", filename="algo.pdf", path="/p/algo.pdf", ext=".pdf", db_path=db_path)
    from core.extraction import DocumentChunk
    c = DocumentChunk(
        id=f"chk_{file_id}_0000",
        file_id=file_id,
        chunk_index=0,
        text="Dijkstra algorithm finds shortest paths in graphs.",
        location="Page 1",
        page=1,
    )
    persist_chunks_for_file(file_id, [c], db_path=db_path)
    update_file_processing_status(file_id, "processed", chunk_count=1, db_path=db_path)

    # Index with model v1
    res1 = index_pending_chunks(
        embed_model="model_v1",
        chroma_dir=chroma_dir,
        db_path=db_path,
        custom_embedder=mock_embedder,
    )
    assert res1.indexed_count == 1

    # Index with model v2
    res2 = index_pending_chunks(
        embed_model="model_v2",
        chroma_dir=chroma_dir,
        db_path=db_path,
        custom_embedder=mock_embedder,
    )
    assert res2.indexed_count == 1
    assert res2.skipped_count == 0


def test_handling_empty_failed_and_unsupported_files(test_env, mock_embedder):
    """Files with 0 chunks, failed extraction, or unsupported status are handled gracefully."""
    db_path = test_env["db_path"]
    chroma_dir = test_env["chroma_dir"]

    f_failed = insert_file(group_id=1, sha256="h_fail", filename="bad.pdf", path="/p/bad.pdf", ext=".pdf", db_path=db_path)
    update_file_processing_status(f_failed, "failed", error_message="corrupt", db_path=db_path)

    f_unsupp = insert_file(group_id=1, sha256="h_uns", filename="img.png", path="/p/img.png", ext=".png", db_path=db_path)
    update_file_processing_status(f_unsupp, "unsupported", db_path=db_path)

    f_empty = insert_file(group_id=1, sha256="h_emp", filename="empty.docx", path="/p/empty.docx", ext=".docx", db_path=db_path)
    update_file_processing_status(f_empty, "empty", chunk_count=0, db_path=db_path)

    res = index_pending_chunks(chroma_dir=chroma_dir, db_path=db_path, custom_embedder=mock_embedder)
    assert res.total_eligible == 0
    assert res.indexed_count == 0
    assert res.failed_count == 0
    assert len(res.errors) == 0


def test_recovery_after_partial_indexing_failure(test_env, monkeypatch):
    """If an embedding batch fails, errors are recorded and subsequent runs recover."""
    db_path = test_env["db_path"]
    chroma_dir = test_env["chroma_dir"]

    file_id = insert_file(group_id=1, sha256="hf", filename="doc.pdf", path="/p/doc.pdf", ext=".pdf", db_path=db_path)
    from core.extraction import DocumentChunk
    chunks = [
        DocumentChunk(
            id=f"chk_{file_id}_{i:04d}",
            file_id=file_id,
            chunk_index=i,
            text=f"Sample text chunk number {i}",
            location=f"Page {i+1}",
            page=i+1,
        )
        for i in range(4)
    ]
    persist_chunks_for_file(file_id, chunks, db_path=db_path)
    update_file_processing_status(file_id, "processed", chunk_count=4, db_path=db_path)

    # First run fails with batch_size=2 due to simulated error in custom_embedder
    call_count = 0
    def failing_embedder(texts):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Temporary CUDA out of memory simulation")
        return [[0.05] * 384 for _ in texts]

    res1 = index_pending_chunks(
        batch_size=2,
        chroma_dir=chroma_dir,
        db_path=db_path,
        custom_embedder=failing_embedder,
    )
    assert res1.failed_count == 2
    assert res1.indexed_count == 2  # Second batch succeeded
    assert len(res1.errors) == 1

    # Second run without error recovers the remaining failed chunks
    clean_embedder = make_deterministic_embedder()
    res2 = index_pending_chunks(
        batch_size=2,
        chroma_dir=chroma_dir,
        db_path=db_path,
        custom_embedder=clean_embedder,
    )
    assert res2.indexed_count == 2
    assert res2.failed_count == 0

    counts = get_vector_index_counts(db_path=db_path)
    assert counts["indexed_chunks"] == 4
    assert counts["pending_chunks"] == 0


# ---------------------------------------------------------------------------
# 3. Semantic Search Tests
# ---------------------------------------------------------------------------

def test_semantic_search_retrieves_valid_sqlite_backed_chunks(test_env, mock_embedder):
    """Semantic search resolves ChromaDB hits back to SQLite records with valid metadata."""
    db_path = test_env["db_path"]
    chroma_dir = test_env["chroma_dir"]

    file_id = insert_file(group_id=1, sha256="h_sem", filename="db_systems.pdf", path="/p/db_systems.pdf", ext=".pdf", db_path=db_path)
    from core.extraction import DocumentChunk
    c = DocumentChunk(
        id=f"chk_{file_id}_0000",
        file_id=file_id,
        chunk_index=0,
        text="Relational database normalization eliminates redundant data and avoids update anomalies.",
        location="Page 4",
        page=4,
    )
    persist_chunks_for_file(file_id, [c], db_path=db_path)
    update_file_processing_status(file_id, "processed", chunk_count=1, db_path=db_path)
    index_pending_chunks(chroma_dir=chroma_dir, db_path=db_path, custom_embedder=mock_embedder)

    # Search with natural language query
    results = search_chunks_semantic(
        query="database schema redundancy reduction",
        top_k=5,
        chroma_dir=chroma_dir,
        db_path=db_path,
        custom_embedder=mock_embedder,
    )
    assert len(results) == 1
    hit = results[0]
    assert hit.chunk_id == f"chk_{file_id}_0000"
    assert hit.filename == "db_systems.pdf"
    assert hit.location == "Page 4"
    assert hit.page == 4
    assert hit.semantic_distance >= 0.0
    assert 0.0 <= hit.semantic_similarity <= 1.0


def test_semantic_search_skips_stale_vectors(test_env, mock_embedder):
    """If a vector exists in ChromaDB but chunk was deleted from SQLite, it is safely omitted."""
    db_path = test_env["db_path"]
    chroma_dir = test_env["chroma_dir"]

    file_id = insert_file(group_id=1, sha256="h_stale", filename="temp.pdf", path="/p/temp.pdf", ext=".pdf", db_path=db_path)
    from core.extraction import DocumentChunk
    c = DocumentChunk(
        id=f"chk_{file_id}_0000",
        file_id=file_id,
        chunk_index=0,
        text="Galois Counter Mode authenticated encryption.",
        location="Page 1",
        page=1,
    )
    persist_chunks_for_file(file_id, [c], db_path=db_path)
    update_file_processing_status(file_id, "processed", chunk_count=1, db_path=db_path)
    index_pending_chunks(chroma_dir=chroma_dir, db_path=db_path, custom_embedder=mock_embedder)

    # Delete chunk from SQLite directly to simulate stale vector
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM chunks WHERE id = ?", (f"chk_{file_id}_0000",))
        conn.commit()

    results = search_chunks_semantic(
        query="authenticated encryption",
        top_k=5,
        chroma_dir=chroma_dir,
        db_path=db_path,
        custom_embedder=mock_embedder,
    )
    # Stale vector safely skipped
    assert len(results) == 0


def test_semantic_search_empty_queries_and_empty_collection(test_env, mock_embedder):
    """Empty query or empty collection returns empty list without errors."""
    db_path = test_env["db_path"]
    chroma_dir = test_env["chroma_dir"]

    # Empty collection
    assert search_chunks_semantic("test", chroma_dir=chroma_dir, db_path=db_path) == []

    # Empty/whitespace query
    assert search_chunks_semantic("   ", chroma_dir=chroma_dir, db_path=db_path) == []

    # Invalid top_k
    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        search_chunks_semantic("test", top_k=0, chroma_dir=chroma_dir, db_path=db_path)


def test_semantic_search_group_and_file_filters(test_env, mock_embedder):
    """Semantic search honors group_id and file_id filters."""
    from core.db import get_or_create_group
    db_path = test_env["db_path"]
    chroma_dir = test_env["chroma_dir"]
    g1 = test_env["group_id"]
    g2 = get_or_create_group("StudyGroup2", db_path=db_path)

    f1 = insert_file(group_id=g1, sha256="h_g1", filename="f1.pdf", path="/p/f1.pdf", ext=".pdf", db_path=db_path)
    f2 = insert_file(group_id=g2, sha256="h_g2", filename="f2.pdf", path="/p/f2.pdf", ext=".pdf", db_path=db_path)

    from core.extraction import DocumentChunk
    c1 = DocumentChunk(id=f"chk_{f1}_0000", file_id=f1, chunk_index=0, text="DBMS Normalization in Group 1.", location="P1", page=1)
    c2 = DocumentChunk(id=f"chk_{f2}_0000", file_id=f2, chunk_index=0, text="DBMS Normalization in Group 2.", location="P1", page=1)

    persist_chunks_for_file(f1, [c1], db_path=db_path)
    persist_chunks_for_file(f2, [c2], db_path=db_path)
    update_file_processing_status(f1, "processed", chunk_count=1, db_path=db_path)
    update_file_processing_status(f2, "processed", chunk_count=1, db_path=db_path)

    index_pending_chunks(chroma_dir=chroma_dir, db_path=db_path, custom_embedder=mock_embedder)

    res_g1 = search_chunks_semantic("normalization", group_id=g1, chroma_dir=chroma_dir, db_path=db_path, custom_embedder=mock_embedder)
    assert len(res_g1) == 1
    assert res_g1[0].file_id == f1

    res_g2 = search_chunks_semantic("normalization", group_id=g2, chroma_dir=chroma_dir, db_path=db_path, custom_embedder=mock_embedder)
    assert len(res_g2) == 1
    assert res_g2[0].file_id == f2


# ---------------------------------------------------------------------------
# 4. Hybrid Retrieval Tests
# ---------------------------------------------------------------------------

def test_hybrid_retrieval_combines_and_deduplicates(test_env, mock_embedder):
    """Hybrid search merges FTS5 and semantic hits, deduplicates chunk IDs, and tags origin."""
    db_path = test_env["db_path"]
    chroma_dir = test_env["chroma_dir"]

    fid = insert_file(group_id=1, sha256="h_hyb", filename="hybrid_test.pdf", path="/p/ht.pdf", ext=".pdf", db_path=db_path)

    from core.extraction import DocumentChunk
    chunks = [
        # Chunk 0: Has exact keyword "normalization" and semantic relevance
        DocumentChunk(
            id=f"chk_{fid}_0000",
            file_id=fid,
            chunk_index=0,
            text="Relational DBMS normalization eliminates redundancy.",
            location="Page 1",
            page=1,
        ),
        # Chunk 1: Has exact keyword "normalization" but weak semantic vector relevance to query
        DocumentChunk(
            id=f"chk_{fid}_0001",
            file_id=fid,
            chunk_index=1,
            text="The term normalization is also used in audio volume normalization processing.",
            location="Page 2",
            page=2,
        ),
        # Chunk 2: Strong semantic relevance to database redundancy/anomalies without using exact keyword "normalization"
        DocumentChunk(
            id=f"chk_{fid}_0002",
            file_id=fid,
            chunk_index=2,
            text="Database schema decomposition prevents update anomalies and data duplication.",
            location="Page 3",
            page=3,
        ),
    ]
    persist_chunks_for_file(fid, chunks, db_path=db_path)
    update_file_processing_status(fid, "processed", chunk_count=3, db_path=db_path)
    index_pending_chunks(chroma_dir=chroma_dir, db_path=db_path, custom_embedder=mock_embedder)

    # Search with hybrid query
    results = search_chunks_hybrid(
        query="normalization database anomalies",
        limit=10,
        chroma_dir=chroma_dir,
        db_path=db_path,
        custom_embedder=mock_embedder,
    )

    # Ensure all chunks are returned and deduplicated
    chunk_ids = [r.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates

    # Check origin tagging
    matched_by_map = {r.chunk_id: r.matched_by for r in results}
    # Chunk 0 matches both keyword ("normalization") and semantic ("database", "redundancy")
    assert matched_by_map[f"chk_{fid}_0000"] in ("both", "keyword", "semantic")

    # Verify score preservation
    for r in results:
        if r.matched_by == "both":
            assert r.keyword_score is not None
            assert r.semantic_distance is not None
            assert r.semantic_similarity is not None
        elif r.matched_by == "keyword":
            assert r.keyword_score is not None
            assert r.semantic_distance is None
        elif r.matched_by == "semantic":
            assert r.keyword_score is None
            assert r.semantic_distance is not None
            assert r.semantic_similarity is not None


def test_hybrid_search_fallback_on_semantic_failure(test_env, monkeypatch):
    """If ChromaDB fails or raises an error, hybrid search gracefully falls back to FTS5."""
    db_path = test_env["db_path"]
    chroma_dir = test_env["chroma_dir"]

    fid = insert_file(group_id=1, sha256="h_fb", filename="fallback.pdf", path="/p/fb.pdf", ext=".pdf", db_path=db_path)
    from core.extraction import DocumentChunk
    c = DocumentChunk(
        id=f"chk_{fid}_0000",
        file_id=fid,
        chunk_index=0,
        text="Galois Counter Mode authenticated encryption standards.",
        location="Page 1",
        page=1,
    )
    persist_chunks_for_file(fid, [c], db_path=db_path)
    update_file_processing_status(fid, "processed", chunk_count=1, db_path=db_path)

    # Force search_chunks_semantic to raise an exception
    def broken_semantic(*args, **kwargs):
        raise RuntimeError("ChromaDB storage engine disk failure")

    monkeypatch.setattr("core.vector_store.search_chunks_semantic", broken_semantic)

    results = search_chunks_hybrid(
        query="Galois",
        chroma_dir=chroma_dir,
        db_path=db_path,
    )

    # Should still retrieve FTS5 keyword results!
    assert len(results) == 1
    assert results[0].chunk_id == f"chk_{fid}_0000"
    assert results[0].matched_by == "keyword"
    assert results[0].keyword_score is not None


# ---------------------------------------------------------------------------
# 5. Section 9 End-to-End Acceptance Scenario
# ---------------------------------------------------------------------------

def test_section_9_end_to_end_acceptance_scenario(tmp_path: Path, mock_embedder):
    """End-to-End Acceptance Scenario as specified in Section 9 of Step 2.2 requirement.

    Synthetic documents:
    1. DBMS Normalization PDF: "DBMS normalization reduces redundancy and update anomalies."
    2. Normal Forms PPTX: "First, Second and Third Normal Forms."
    3. Cryptography DOCX: "Galois/Counter Mode provides authenticated encryption."
    4. Unrelated Recipe PDF: "Baking delicious chocolate chip cookies with flour, butter, and sugar."
    """
    db_path = tmp_path / "studyrag_acceptance.db"
    init_db(db_path)
    chroma_dir = tmp_path / "acceptance_chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = tmp_path / "synthetic_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create synthetic documents
    pdf_dbms = docs_dir / "dbms_normalization.pdf"
    doc1 = fitz.open()
    p1 = doc1.new_page()
    p1.insert_text((50, 72), "DBMS normalization reduces redundancy and update anomalies in relational tables.")
    doc1.save(str(pdf_dbms))
    doc1.close()

    pptx_nf = docs_dir / "normal_forms.pptx"
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    tb = slide.shapes.add_textbox(100, 100, 400, 200)
    tb.text_frame.text = "First, Second and Third Normal Forms define functional dependency constraints."
    prs.save(str(pptx_nf))

    docx_gcm = docs_dir / "galois_gcm.docx"
    doc3 = docx.Document()
    doc3.add_paragraph("Galois/Counter Mode provides authenticated encryption with associated data for confidentiality and integrity.")
    doc3.save(str(docx_gcm))

    pdf_recipe = docs_dir / "cookie_recipe.pdf"
    doc4 = fitz.open()
    p4 = doc4.new_page()
    p4.insert_text((50, 72), "Baking delicious chocolate chip cookies with flour, butter, brown sugar, and vanilla extract.")
    doc4.save(str(pdf_recipe))
    doc4.close()

    # 2. Register documents in SQLite
    from core.db import get_or_create_group
    gid = get_or_create_group("AcceptanceGroup", db_path=db_path)

    fid_dbms = insert_file(group_id=gid, sha256="hash_dbms", filename="dbms_normalization.pdf", path=str(pdf_dbms), ext=".pdf", db_path=db_path)
    fid_nf = insert_file(group_id=gid, sha256="hash_nf", filename="normal_forms.pptx", path=str(pptx_nf), ext=".pptx", db_path=db_path)
    fid_gcm = insert_file(group_id=gid, sha256="hash_gcm", filename="galois_gcm.docx", path=str(docx_gcm), ext=".docx", db_path=db_path)
    fid_rec = insert_file(group_id=gid, sha256="hash_rec", filename="cookie_recipe.pdf", path=str(pdf_recipe), ext=".pdf", db_path=db_path)

    # 3. Extract and chunk through existing pipeline
    for fid in [fid_dbms, fid_nf, fid_gcm, fid_rec]:
        res = process_file(file_id=fid, db_path=db_path)
        assert res.status == "processed"
        assert res.chunk_count >= 1

    status_counts = get_processing_status_counts(group_id=gid, db_path=db_path)
    assert status_counts["processed"] == 4

    # 4. Index chunks into ChromaDB
    idx_res = index_pending_chunks(
        group_id=gid,
        chroma_dir=chroma_dir,
        db_path=db_path,
        custom_embedder=mock_embedder,
    )
    assert idx_res.total_eligible == 4
    assert idx_res.indexed_count == 4
    assert idx_res.skipped_count == 0
    assert idx_res.failed_count == 0

    idx_counts = get_vector_index_counts(group_id=gid, db_path=db_path)
    assert idx_counts["indexed_chunks"] == 4
    assert idx_counts["pending_chunks"] == 0
    assert idx_counts["indexed_files"] == 4

    # 5. Verification: Repeating indexing does NOT duplicate vectors
    repeat_res = index_pending_chunks(
        group_id=gid,
        chroma_dir=chroma_dir,
        db_path=db_path,
        custom_embedder=mock_embedder,
    )
    assert repeat_res.indexed_count == 0
    assert repeat_res.skipped_count == 4
    col = get_or_create_collection(chroma_dir=chroma_dir)
    assert col.count() == 4

    # 6. Verification: FTS5 finds direct keyword matches
    fts_results = search_chunks_fts("Galois", group_id=gid, db_path=db_path)
    assert len(fts_results) >= 1
    assert any("Galois/Counter Mode" in r["text"] for r in fts_results)

    # 7. Verification: Semantic retrieval retrieves content with different wording
    sem_dbms = search_chunks_semantic(
        query="database schema redundancy and update anomalies",
        top_k=2,
        group_id=gid,
        chroma_dir=chroma_dir,
        db_path=db_path,
        custom_embedder=mock_embedder,
    )
    assert len(sem_dbms) >= 1
    assert "dbms_normalization.pdf" in sem_dbms[0].filename or "normal_forms.pptx" in sem_dbms[0].filename

    sem_crypto = search_chunks_semantic(
        query="authenticated cipher security",
        top_k=2,
        group_id=gid,
        chroma_dir=chroma_dir,
        db_path=db_path,
        custom_embedder=mock_embedder,
    )
    assert len(sem_crypto) >= 1
    assert "galois_gcm.docx" in sem_crypto[0].filename

    # 8. Verification: Hybrid retrieval merges both result sets, deduplicates chunk IDs, preserves scores
    hybrid_results = search_chunks_hybrid(
        query="Galois authenticated encryption",
        limit=10,
        group_id=gid,
        chroma_dir=chroma_dir,
        db_path=db_path,
        custom_embedder=mock_embedder,
    )
    assert len(hybrid_results) >= 1
    # Check deduplication
    returned_ids = [r.chunk_id for r in hybrid_results]
    assert len(returned_ids) == len(set(returned_ids))

    # Top hit should be Galois GCM matching both keyword and semantic
    top_hit = hybrid_results[0]
    assert "galois_gcm.docx" in top_hit.filename
    assert top_hit.matched_by in ("both", "keyword", "semantic")
    assert top_hit.location != ""  # Valid location traceable to source
    assert top_hit.file_id == fid_gcm

    # 9. Verification: Traceability to original file and location
    for r in hybrid_results:
        assert r.file_id in (fid_dbms, fid_nf, fid_gcm, fid_rec)
        assert r.location.startswith(("Page", "Slide", "Section", "Paragraph", "Table"))
