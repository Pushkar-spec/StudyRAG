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
    at = AppTest.from_file(str(app_path), default_timeout=60).run(timeout=60)
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

    # Step 4.1 WhatsApp context symbols
    assert callable(app.get_whatsapp_context_for_documents)
    assert callable(app.get_available_senders)
    assert callable(app.get_available_date_range)
    assert callable(app.WhatsAppFilterCriteria)

    # Step 4.2 WhatsApp context-aware ranking symbols
    assert callable(app.score_whatsapp_context)
    assert callable(app.apply_context_ranking)
    assert callable(app.WhatsAppContextScore)
    assert isinstance(app.DEFAULT_CONTEXT_WEIGHT, float)


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


def test_attachment_recovery_renders_with_sqlite_rows_without_pickling_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify that AppTest renders successfully when groups, references, and files exist in SQLite."""
    monkeypatch.setenv("STUDYRAG_HOME", str(tmp_path))
    db_path = tmp_path / "studyrag.db"
    init_db(db_path)

    from core.db import get_or_create_group
    gid = get_or_create_group("Physics_101", db_path=db_path)

    upsert_attachment_ref(
        group_id=gid,
        dedup_key="key_phys_1",
        status="missing",
        original_filename="Electromagnetism.pdf",
        db_path=db_path,
    )
    insert_file(
        group_id=gid,
        sha256="hash_phys_1",
        filename="Electromagnetism.pdf",
        path=str(tmp_path / "Electromagnetism.pdf"),
        ext=".pdf",
        db_path=db_path,
    )

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=60).run(timeout=60)

    # Must run cleanly without TypeError: cannot pickle 'sqlite3.Row' object
    assert not at.exception

    # Group selectbox should be present and formatted with human-readable labels
    rec_group_sb = at.selectbox(key="rec_group_select")
    assert rec_group_sb is not None
    assert rec_group_sb.options == [f"Physics_101 (ID: {gid})"]
    assert rec_group_sb.value == gid


def test_group_selection_options_are_primitives_not_sqlite_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Ensure all group selectboxes use primitive integer IDs or None as options."""
    monkeypatch.setenv("STUDYRAG_HOME", str(tmp_path))
    db_path = tmp_path / "studyrag.db"
    init_db(db_path)

    from core.db import get_or_create_group
    g1 = get_or_create_group("Group Alpha", db_path=db_path)
    g2 = get_or_create_group("Group Beta", db_path=db_path)

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=60).run(timeout=60)
    assert not at.exception

    # Filter group select in recovery
    rec_filter = at.selectbox(key="rec_filter_group_id")
    assert rec_filter is not None
    assert rec_filter.options == ["All Groups", f"Group Alpha (ID: {g1})", f"Group Beta (ID: {g2})"]

    # Target group select in recovery tab 1
    rec_group = at.selectbox(key="rec_group_select")
    assert rec_group is not None
    assert rec_group.options == [f"Group Alpha (ID: {g1})", f"Group Beta (ID: {g2})"]

    # Document processing group select
    doc_proc_group = at.selectbox(key="doc_proc_group_select")
    assert doc_proc_group is not None
    assert doc_proc_group.options == ["All Groups", f"Group Alpha (ID: {g1})", f"Group Beta (ID: {g2})"]

    # Vector indexing group select
    vec_group = at.selectbox(key="vec_group_select")
    assert vec_group is not None
    assert vec_group.options == ["All Groups", f"Group Alpha (ID: {g1})", f"Group Beta (ID: {g2})"]


def test_section_isolation_prevents_cascade_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An unexpected exception in Attachment Recovery does not prevent Document Processing and Vector Indexing from rendering."""
    monkeypatch.setenv("STUDYRAG_HOME", str(tmp_path))
    db_path = tmp_path / "studyrag.db"
    init_db(db_path)

    from core.db import get_or_create_group
    get_or_create_group("Math_101", db_path=db_path)

    def failing_get_attachment_ref_counts(*args, **kwargs):
        raise RuntimeError("Simulated attachment recovery failure")

    monkeypatch.setattr("core.db.get_attachment_ref_counts", failing_get_attachment_ref_counts)

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=60).run(timeout=60)

    # Top-level script must not fail with unhandled exception
    assert not at.exception

    # An error message should be displayed for the failed section
    errors = [e.value for e in at.error]
    assert any("Attachment Recovery section encountered an unexpected error" in e for e in errors)
    assert any("Simulated attachment recovery failure" in e for e in errors)

    # Document Processing and Vector Indexing sections should still be rendered
    subheaders = [s.value for s in at.subheader]
    assert any("Document Processing" in s for s in subheaders)
    assert any("Vector Indexing" in s for s in subheaders)


def test_attachment_recovery_manual_linking_with_primitive_ids(tmp_path: Path):
    """Verify that manual linking works with primitive integer IDs and updates attachment ref status."""
    db_path = tmp_path / "studyrag.db"
    init_db(db_path)

    from core.db import get_or_create_group, get_attachment_ref
    gid = get_or_create_group("CS_101", db_path=db_path)

    ref_id = upsert_attachment_ref(
        group_id=gid,
        dedup_key="key_cs1",
        status="missing",
        original_filename="Algorithm_Notes.pdf",
        db_path=db_path,
    )

    file_id = insert_file(
        group_id=gid,
        sha256="hash_cs1",
        filename="Algorithm_Notes.pdf",
        path=str(tmp_path / "Algorithm_Notes.pdf"),
        ext=".pdf",
        db_path=db_path,
    )

    ok, msg = link_attachment_ref_manually(ref_id=ref_id, file_id=file_id, db_path=db_path)
    assert ok is True
    assert "linked" in msg.lower()

    ref = get_attachment_ref(ref_id, db_path=db_path)
    assert ref is not None
    assert ref["status"] == "available"
    assert ref["file_id"] == file_id
    assert ref["resolution_method"] == "manual"


def test_ui_ranked_documents_search_mode_and_render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that search mode selectbox includes Ranked Documents and renders without errors."""
    monkeypatch.setenv("STUDYRAG_HOME", str(tmp_path))
    db_path = tmp_path / "studyrag.db"
    init_db(db_path)

    from core.db import get_or_create_group
    get_or_create_group("CS_101", db_path=db_path)

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=60).run(timeout=60)
    assert not at.exception

    search_mode_sb = at.selectbox(key="hybrid_mode_select")
    assert search_mode_sb is not None
    assert "Ranked Documents (Hybrid + Cross-Encoder)" in search_mode_sb.options
    assert search_mode_sb.value == "Ranked Documents (Hybrid + Cross-Encoder)"


def test_ui_whatsapp_context_filter_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that WhatsApp context filter widgets render properly in AppTest."""
    monkeypatch.setenv("STUDYRAG_HOME", str(tmp_path))
    db_path = tmp_path / "studyrag.db"
    init_db(db_path)

    from core.db import get_or_create_group, insert_file, upsert_attachment_ref
    gid = get_or_create_group("CS_101", db_path=db_path)
    f_id = insert_file(
        group_id=gid,
        sha256="hash123",
        filename="lecture1.pdf",
        path=str(tmp_path / "lecture1.pdf"),
        ext=".pdf",
        sender="Prof. Alan",
        sent_at="2026-03-15T10:00:00",
        caption="Review Chapter 1 notes",
        db_path=db_path,
    )
    upsert_attachment_ref(
        group_id=gid,
        dedup_key="key_lec1",
        status="available",
        file_id=f_id,
        sender="Prof. Alan",
        sent_at="2026-03-15T10:00:00",
        caption="Review Chapter 1 notes",
        context="[2026-03-15 09:58] Prof. Alan: Today we start.\n[2026-03-15 10:00] Prof. Alan: Review Chapter 1 notes",
        original_filename="lecture1.pdf",
        resolution_method="direct",
        db_path=db_path,
    )

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=60).run(timeout=60)
    assert not at.exception

    # Check that WhatsApp context filter widgets exist
    require_linked_cb = at.checkbox(key="wa_filter_require_linked")
    assert require_linked_cb is not None
    assert require_linked_cb.value is False

    enable_ctx_cb = at.checkbox(key="wa_enable_context_ranking")
    assert enable_ctx_cb is not None
    assert enable_ctx_cb.value is True

    senders_ms = at.multiselect(key="wa_filter_senders")
    assert senders_ms is not None
    assert "Prof. Alan" in senders_ms.options

    date_from_input = at.text_input(key="wa_filter_date_from")
    assert date_from_input is not None

    date_to_input = at.text_input(key="wa_filter_date_to")
    assert date_to_input is not None

    msg_query_input = at.text_input(key="wa_filter_msg_query")
    assert msg_query_input is not None

