"""UI and Streamlit integration tests for StudyRAG (Step 1.3)."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from core.db import (
    get_all_groups,
    get_attachment_ref,
    get_attachment_ref_counts,
    get_attachment_refs,
    init_db,
    insert_file,
    upsert_attachment_ref,
)
from core.ingest import link_attachment_ref_manually, recover_file_or_folder


def test_app_launches_without_exceptions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that app.py compiles, initializes, and renders without exceptions."""
    monkeypatch.setenv("STUDYRAG_HOME", str(tmp_path))
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30).run(timeout=30)
    assert not at.exception
    assert len(at.title) > 0 or len(at.markdown) > 0


def test_ui_attachment_counts_reflect_database(tmp_path: Path):
    """Attachment reference counts accurately reflect database state."""
    db_path = tmp_path / "test.db"
    init_db(db_path)

    # Insert test group and references
    from core.db import get_or_create_group
    group_id = get_or_create_group("CS101", db_path=db_path)

    upsert_attachment_ref(
        group_id=group_id,
        dedup_key="key1",
        status="missing",
        original_filename="notes.pdf",
        db_path=db_path,
    )
    upsert_attachment_ref(
        group_id=group_id,
        dedup_key="key2",
        status="available",
        original_filename="syllabus.pdf",
        db_path=db_path,
    )
    upsert_attachment_ref(
        group_id=group_id,
        dedup_key="key3",
        status="ambiguous",
        original_filename="quiz.pdf",
        db_path=db_path,
    )

    counts = get_attachment_ref_counts(group_id=group_id, db_path=db_path)
    assert counts["total"] == 3
    assert counts["missing"] == 1
    assert counts["available"] == 1
    assert counts["ambiguous"] == 1
    assert counts["unsupported"] == 0
    assert counts["import_failed"] == 0


def test_invalid_or_cross_group_linking_rejected_safely(tmp_path: Path):
    """Manual linking between a reference in Group A and file in Group B is rejected."""
    db_path = tmp_path / "test.db"
    init_db(db_path)

    from core.db import get_or_create_group
    g1 = get_or_create_group("GroupA", db_path=db_path)
    g2 = get_or_create_group("GroupB", db_path=db_path)

    ref_id = upsert_attachment_ref(
        group_id=g1,
        dedup_key="keyA",
        status="missing",
        original_filename="docA.pdf",
        db_path=db_path,
    )

    file_id = insert_file(
        group_id=g2,
        sha256="hashB",
        filename="docA.pdf",
        path="/some/path/docA.pdf",
        ext=".pdf",
        db_path=db_path,
    )

    ok, msg = link_attachment_ref_manually(ref_id=ref_id, file_id=file_id, db_path=db_path)
    assert ok is False
    assert "Cross-group linking disallowed" in msg

    # Non-existent reference
    ok2, msg2 = link_attachment_ref_manually(ref_id=9999, file_id=file_id, db_path=db_path)
    assert ok2 is False
    assert "not found" in msg2


def test_app_all_symbols_and_imports_resolve():
    """Regression test: verify app.py imports cleanly without missing or invalid symbols."""
    import app
    assert callable(app.initialize_foundation)
    assert callable(app.render_ingestion_preview)
    assert callable(app.render_attachment_recovery)
    assert callable(app.render_document_processing)
    assert callable(app.main)

    # Verify functions imported from core.ingest, core.extraction, and core.db exist and are callable
    assert callable(app.ingest_zip)
    assert callable(app.ingest_folder)
    assert callable(app.recover_file_or_folder)
    assert callable(app.link_attachment_ref_manually)
    assert callable(app.reconcile_attachment_refs)
    assert callable(app.get_candidates_for_ref)
    assert callable(app.get_attachment_refs)
    assert callable(app.get_attachment_ref_counts)
    assert callable(app.process_file)
    assert callable(app.process_group_files)
    assert callable(app.get_processing_status_counts)
    assert callable(app.get_file_chunks)

    # Step 2.2 vector indexing and search symbols
    assert callable(app.render_vector_indexing_and_search)
    assert callable(app.get_vector_index_counts)
    assert callable(app.index_pending_chunks)
    assert callable(app.search_chunks_hybrid)
    assert callable(app.search_chunks_semantic)


def test_ui_document_processing_status_counts(tmp_path: Path):
    """File processing status counts reflect actual SQLite states."""
    db_path = tmp_path / "test.db"
    init_db(db_path)

    from core.db import get_or_create_group, update_file_processing_status, get_processing_status_counts
    gid = get_or_create_group("OS_Group", db_path=db_path)

    f1 = insert_file(gid, "h1", "f1.pdf", "/p1", ".pdf", db_path=db_path)
    f2 = insert_file(gid, "h2", "f2.docx", "/p2", ".docx", db_path=db_path)
    f3 = insert_file(gid, "h3", "f3.png", "/p3", ".png", db_path=db_path)

    update_file_processing_status(f1, "processed", chunk_count=5, db_path=db_path)
    update_file_processing_status(f2, "failed", error_message="corrupt", db_path=db_path)
    update_file_processing_status(f3, "unsupported", db_path=db_path)

    counts = get_processing_status_counts(group_id=gid, db_path=db_path)
    assert counts["total"] == 3
    assert counts["processed"] == 1
    assert counts["failed"] == 1
    assert counts["unsupported"] == 1
    assert counts["empty"] == 0
    assert counts["new"] == 0


def test_ui_vector_index_status_counts(tmp_path: Path):
    """Vector indexing status counts accurately reflect SQLite vector state."""
    db_path = tmp_path / "test.db"
    init_db(db_path)

    from core.db import get_or_create_group, update_file_processing_status, get_vector_index_counts, record_chunks_indexed_batch, update_file_indexing_status
    from core.extraction import DocumentChunk, persist_chunks_for_file
    gid = get_or_create_group("AI_Group", db_path=db_path)

    f1 = insert_file(gid, "h1", "f1.pdf", "/p1", ".pdf", db_path=db_path)
    c1 = DocumentChunk(id=f"chk_{f1}_0000", file_id=f1, chunk_index=0, text="AI intro.", location="P1", page=1)
    persist_chunks_for_file(f1, [c1], db_path=db_path)
    update_file_processing_status(f1, "processed", chunk_count=1, db_path=db_path)

    # Initial counts: 1 eligible chunk, 0 indexed, 1 pending
    counts1 = get_vector_index_counts(group_id=gid, db_path=db_path)
    assert counts1["eligible_chunks"] == 1
    assert counts1["indexed_chunks"] == 0
    assert counts1["pending_chunks"] == 1
    assert counts1["indexed_files"] == 0

    # Record indexed
    record_chunks_indexed_batch([(f"chk_{f1}_0000", f1, "BAAI/bge-small-en-v1.5", "some_hash")], db_path=db_path)
    update_file_indexing_status(f1, db_path=db_path)

    counts2 = get_vector_index_counts(group_id=gid, db_path=db_path)
    assert counts2["eligible_chunks"] == 1
    assert counts2["indexed_chunks"] == 1
    assert counts2["pending_chunks"] == 0
    assert counts2["indexed_files"] == 1

