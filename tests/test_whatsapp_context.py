"""Tests for WhatsApp context retrieval and context-aware filtering (Phase 4, Step 4.1)."""

from pathlib import Path
import pytest

from core.db import (
    db_session,
    get_or_create_group,
    init_db,
    insert_file,
    upsert_attachment_ref,
)
from core.document_ranking import (
    RankedDocument,
    SearchRankedResult,
    search_ranked_documents,
)
from core.whatsapp_context import (
    DocumentWhatsAppContext,
    FilterEvaluationResult,
    WhatsAppFilterCriteria,
    WhatsAppMessageContext,
    evaluate_document_filter,
    filter_ranked_documents,
    get_available_date_range,
    get_available_senders,
    get_whatsapp_context_for_documents,
    get_whatsapp_context_for_file,
)


@pytest.fixture
def context_db(tmp_path: Path):
    """Setup isolated SQLite database with synthetic files, groups, and attachment references."""
    db_path = tmp_path / "context_test.db"
    init_db(db_path)

    # 1. Groups
    gid1 = get_or_create_group("CS_101_DBMS", db_path=db_path)
    gid2 = get_or_create_group("CS_102_Networks", db_path=db_path)

    # 2. Files
    # File 1: Full WhatsApp context (Group 1, Prof. Alan)
    f1 = insert_file(
        group_id=gid1,
        sha256="sha_doc1",
        filename="DBMS_Lecture_01.pdf",
        path="/docs/DBMS_Lecture_01.pdf",
        ext=".pdf",
        sender="Prof. Alan",
        sent_at="2026-03-10T10:30:00",
        caption="Please review normalization before Monday's quiz.\nDBMS_Lecture_01.pdf (file attached)",
        db_path=db_path,
    )

    # File 2: Full WhatsApp context (Group 1, TA Bob, later date)
    f2 = insert_file(
        group_id=gid1,
        sha256="sha_doc2",
        filename="DBMS_Tutorial_02.docx",
        path="/docs/DBMS_Tutorial_02.docx",
        ext=".docx",
        sender="TA Bob",
        sent_at="2026-03-25T14:15:00",
        caption="Tutorial exercises on 3NF and BCNF decomposition.\nDBMS_Tutorial_02.docx (file attached)",
        db_path=db_path,
    )

    # File 3: Standalone file without WhatsApp chat (Group 2, no sender, no timestamp)
    f3 = insert_file(
        group_id=gid2,
        sha256="sha_doc3",
        filename="Unlinked_Networking_Lab.pdf",
        path="/docs/Unlinked_Networking_Lab.pdf",
        ext=".pdf",
        sender=None,
        sent_at=None,
        caption=None,
        db_path=db_path,
    )

    # File 4: File in Group 2 with direct file record but no attachment_refs entry
    f4 = insert_file(
        group_id=gid2,
        sha256="sha_doc4",
        filename="Direct_Record_Only.pdf",
        path="/docs/Direct_Record_Only.pdf",
        ext=".pdf",
        sender="Dr. Charlie",
        sent_at="2026-04-05T09:00:00",
        caption="Direct record caption from folder import.",
        db_path=db_path,
    )

    # 3. Attachment References
    # Ref 1 for File 1
    upsert_attachment_ref(
        group_id=gid1,
        dedup_key="ref_doc1_a",
        status="available",
        original_filename="DBMS_Lecture_01.pdf",
        normalized_filename="dbms_lecture_01.pdf",
        sender="Prof. Alan",
        sent_at="2026-03-10T10:30:00",
        caption="DBMS_Lecture_01.pdf (file attached)",
        context="Good morning everyone.\nPlease review normalization before Monday's quiz.\nDBMS_Lecture_01.pdf (file attached)",
        file_id=f1,
        resolution_method="automatic",
        db_path=db_path,
    )

    # Ref 2 for File 1 (second reference: student re-sharing the file)
    upsert_attachment_ref(
        group_id=gid1,
        dedup_key="ref_doc1_b",
        status="available",
        original_filename="DBMS_Lecture_01.pdf",
        normalized_filename="dbms_lecture_01.pdf",
        sender="Alice Student",
        sent_at="2026-03-12T16:00:00",
        caption="Here is the lecture 1 PDF again for anyone who missed it.",
        context="Here is the lecture 1 PDF again for anyone who missed it.",
        file_id=f1,
        resolution_method="manual",
        db_path=db_path,
    )

    # Ref 3 for File 2
    upsert_attachment_ref(
        group_id=gid1,
        dedup_key="ref_doc2_a",
        status="available",
        original_filename="DBMS_Tutorial_02.docx",
        normalized_filename="dbms_tutorial_02.docx",
        sender="TA Bob",
        sent_at="2026-03-25T14:15:00",
        caption="DBMS_Tutorial_02.docx (file attached)",
        context="Tutorial exercises on 3NF and BCNF decomposition.\nDBMS_Tutorial_02.docx (file attached)",
        file_id=f2,
        resolution_method="automatic",
        db_path=db_path,
    )

    return {
        "db_path": db_path,
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "f4": f4,
        "gid1": gid1,
        "gid2": gid2,
    }


def test_get_whatsapp_context_single_and_multiple_messages(context_db):
    """Verify retrieval of sharing message, sender, timestamp, and surrounding context."""
    db_path = context_db["db_path"]
    f1 = context_db["f1"]
    f2 = context_db["f2"]

    contexts = get_whatsapp_context_for_documents([f1, f2], db_path=db_path)
    assert len(contexts) == 2

    # File 1 has multiple linked messages (Prof. Alan and Alice)
    ctx1 = contexts[f1]
    assert ctx1.file_id == f1
    assert ctx1.filename == "DBMS_Lecture_01.pdf"
    assert ctx1.group_name == "CS_101_DBMS"
    assert ctx1.has_linked_message is True
    assert ctx1.sender == "Prof. Alan"
    assert ctx1.sent_at == "2026-03-10T10:30:00"
    assert "DBMS_Lecture_01.pdf" in ctx1.message_text
    assert "Good morning everyone" in ctx1.surrounding_context
    assert len(ctx1.all_messages) == 2
    assert ctx1.all_messages[1].sender == "Alice Student"

    # File 2 has single linked message
    ctx2 = contexts[f2]
    assert ctx2.file_id == f2
    assert ctx2.sender == "TA Bob"
    assert ctx2.sent_at == "2026-03-25T14:15:00"
    assert "BCNF" in ctx2.surrounding_context
    assert len(ctx2.all_messages) == 1


def test_get_whatsapp_context_unlinked_file(context_db):
    """Verify that a document without chat linkage returns clean unlinked context without invented data."""
    db_path = context_db["db_path"]
    f3 = context_db["f3"]

    ctx3 = get_whatsapp_context_for_file(f3, db_path=db_path)
    assert ctx3 is not None
    assert ctx3.file_id == f3
    assert ctx3.has_linked_message is False
    assert ctx3.sender is None
    assert ctx3.sent_at is None
    assert ctx3.message_text is None
    assert ctx3.surrounding_context is None
    assert len(ctx3.all_messages) == 0


def test_get_whatsapp_context_direct_record_fallback(context_db):
    """Verify fallback to files table when attachment_refs entry is missing."""
    db_path = context_db["db_path"]
    f4 = context_db["f4"]

    ctx4 = get_whatsapp_context_for_file(f4, db_path=db_path)
    assert ctx4 is not None
    assert ctx4.file_id == f4
    assert ctx4.has_linked_message is True
    assert ctx4.sender == "Dr. Charlie"
    assert ctx4.sent_at == "2026-04-05T09:00:00"
    assert "Direct record caption" in ctx4.message_text


def test_filter_by_group(context_db):
    """Verify filtering documents by selected WhatsApp groups."""
    db_path = context_db["db_path"]
    f1 = context_db["f1"]
    f3 = context_db["f3"]
    gid1 = context_db["gid1"]
    gid2 = context_db["gid2"]

    contexts = get_whatsapp_context_for_documents([f1, f3], db_path=db_path)

    # Filter allowing only Group 1
    c1 = WhatsAppFilterCriteria(group_ids=[gid1])
    res1 = evaluate_document_filter(contexts[f1], c1)
    res3 = evaluate_document_filter(contexts[f3], c1)
    assert res1.passed is True
    assert res3.passed is False
    assert "not in selected groups" in res3.reason

    # Filter allowing both groups
    c_both = WhatsAppFilterCriteria(group_ids=[gid1, gid2])
    assert evaluate_document_filter(contexts[f1], c_both).passed is True
    assert evaluate_document_filter(contexts[f3], c_both).passed is True


def test_filter_by_sender_and_missing_metadata(context_db):
    """Verify sender matching, case insensitivity, and explicit missing-metadata handling."""
    db_path = context_db["db_path"]
    f1 = context_db["f1"]  # Senders: Prof. Alan, Alice Student
    f2 = context_db["f2"]  # Sender: TA Bob
    f3 = context_db["f3"]  # Sender: None

    contexts = get_whatsapp_context_for_documents([f1, f2, f3], db_path=db_path)

    criteria = WhatsAppFilterCriteria(senders=["prof. alan"])

    # 1. Matching sender (case-insensitive)
    r1 = evaluate_document_filter(contexts[f1], criteria)
    assert r1.passed is True
    assert r1.is_missing_metadata is False

    # 2. Non-matching sender (Bob != Alan)
    r2 = evaluate_document_filter(contexts[f2], criteria)
    assert r2.passed is False
    assert r2.is_missing_metadata is False
    assert "does not match sender filter" in r2.reason

    # 3. Missing sender metadata (unlinked file)
    r3 = evaluate_document_filter(contexts[f3], criteria)
    assert r3.passed is False
    assert r3.is_missing_metadata is True
    assert "has no sender metadata" in r3.reason


def test_filter_by_date_range(context_db):
    """Verify date-range filtering with ISO timestamps and boundary checks."""
    db_path = context_db["db_path"]
    f1 = context_db["f1"]  # 2026-03-10
    f2 = context_db["f2"]  # 2026-03-25
    f3 = context_db["f3"]  # No timestamp

    contexts = get_whatsapp_context_for_documents([f1, f2, f3], db_path=db_path)

    # Date range 2026-03-01 to 2026-03-15
    c_early = WhatsAppFilterCriteria(date_from="2026-03-01", date_to="2026-03-15")
    assert evaluate_document_filter(contexts[f1], c_early).passed is True
    assert evaluate_document_filter(contexts[f2], c_early).passed is False

    # Date range after 2026-03-20
    c_late = WhatsAppFilterCriteria(date_from="2026-03-20")
    assert evaluate_document_filter(contexts[f1], c_late).passed is False
    assert evaluate_document_filter(contexts[f2], c_late).passed is True

    # Missing timestamp on unlinked file
    r3 = evaluate_document_filter(contexts[f3], c_early)
    assert r3.passed is False
    assert r3.is_missing_metadata is True
    assert "has no timestamp metadata" in r3.reason


def test_filter_by_linked_message_requirement(context_db):
    """Verify filtering based on whether a document has a linked sharing message."""
    db_path = context_db["db_path"]
    f1 = context_db["f1"]
    f3 = context_db["f3"]

    contexts = get_whatsapp_context_for_documents([f1, f3], db_path=db_path)

    # Require linked message
    c_req = WhatsAppFilterCriteria(has_linked_message=True)
    assert evaluate_document_filter(contexts[f1], c_req).passed is True
    assert evaluate_document_filter(contexts[f3], c_req).passed is False

    # Require unlinked message
    c_unlinked = WhatsAppFilterCriteria(has_linked_message=False)
    assert evaluate_document_filter(contexts[f1], c_unlinked).passed is False
    assert evaluate_document_filter(contexts[f3], c_unlinked).passed is True


def test_filter_by_message_query(context_db):
    """Verify keyword filtering on WhatsApp message and surrounding conversation text."""
    db_path = context_db["db_path"]
    f1 = context_db["f1"]  # Has "quiz" in context
    f2 = context_db["f2"]  # Has "decomposition" in context
    f3 = context_db["f3"]  # No message

    contexts = get_whatsapp_context_for_documents([f1, f2, f3], db_path=db_path)

    c_quiz = WhatsAppFilterCriteria(message_query="quiz")
    assert evaluate_document_filter(contexts[f1], c_quiz).passed is True
    assert evaluate_document_filter(contexts[f2], c_quiz).passed is False
    assert evaluate_document_filter(contexts[f3], c_quiz).passed is False

    c_decomp = WhatsAppFilterCriteria(message_query="decomposition")
    assert evaluate_document_filter(contexts[f1], c_decomp).passed is False
    assert evaluate_document_filter(contexts[f2], c_decomp).passed is True


def test_parameterized_query_safety(context_db):
    """Verify that malicious strings in sender/date/group inputs are handled safely without SQL injection."""
    db_path = context_db["db_path"]
    f1 = context_db["f1"]

    contexts = get_whatsapp_context_for_documents([f1], db_path=db_path)

    # SQL injection payload in sender and query
    c_malicious = WhatsAppFilterCriteria(
        senders=["' OR 1=1 --"],
        message_query="'; DROP TABLE files; --",
        date_from="2026-01-01' OR '1'='1",
    )

    eval_res = evaluate_document_filter(contexts[f1], c_malicious)
    assert eval_res.passed is False

    # Verify database tables remain intact
    with db_session(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 4


def test_filter_ranked_documents_and_rank_reassignment(context_db):
    """Verify filter_ranked_documents filters candidate documents and re-indexes ranks."""
    db_path = context_db["db_path"]
    f1 = context_db["f1"]
    f2 = context_db["f2"]
    f3 = context_db["f3"]

    doc1 = RankedDocument(file_id=f1, filename="Doc1", doc_type=".pdf", group_id=1, group_name="G1", document_score=9.5, rank=1)
    doc2 = RankedDocument(file_id=f2, filename="Doc2", doc_type=".docx", group_id=1, group_name="G1", document_score=8.5, rank=2)
    doc3 = RankedDocument(file_id=f3, filename="Doc3", doc_type=".pdf", group_id=2, group_name="G2", document_score=7.5, rank=3)

    criteria = WhatsAppFilterCriteria(senders=["TA Bob"])

    surviving, excluded = filter_ranked_documents([doc1, doc2, doc3], criteria, db_path=db_path)
    assert len(surviving) == 1
    assert surviving[0].file_id == f2
    assert len(excluded) == 2


def test_get_available_senders_and_date_bounds(context_db):
    """Verify discovery helpers for available senders and date ranges."""
    db_path = context_db["db_path"]
    gid1 = context_db["gid1"]

    senders_all = get_available_senders(db_path=db_path)
    assert "Prof. Alan" in senders_all
    assert "TA Bob" in senders_all
    assert "Alice Student" in senders_all

    senders_g1 = get_available_senders(group_id=gid1, db_path=db_path)
    assert "Prof. Alan" in senders_g1
    assert "Dr. Charlie" not in senders_g1

    d_min, d_max = get_available_date_range(db_path=db_path)
    assert d_min is not None and d_max is not None
    assert d_min.startswith("2026-03-10")
    assert d_max.startswith("2026-04-05")


def test_search_ranked_documents_with_whatsapp_filters(tmp_path: Path):
    """Verify end-to-end integration: search_ranked_documents with active WhatsApp context filtering."""
    db_path = tmp_path / "integration_test.db"
    init_db(db_path)

    gid = get_or_create_group("DBMS_Group", db_path=db_path)

    # Document A: About Normalization (Prof. Alan)
    fa = insert_file(
        group_id=gid,
        sha256="sha_norm_a",
        filename="Normalization_Guide.pdf",
        path="/docs/norm_a.pdf",
        ext=".pdf",
        sender="Prof. Alan",
        sent_at="2026-03-01T09:00:00",
        caption="Study guide for 1NF, 2NF and 3NF normalization.\nNormalization_Guide.pdf (file attached)",
        status="processed",
        db_path=db_path,
    )

    # Document B: About Normalization (Alice Student, different date)
    fb = insert_file(
        group_id=gid,
        sha256="sha_norm_b",
        filename="Normalization_Summary.pdf",
        path="/docs/norm_b.pdf",
        ext=".pdf",
        sender="Alice Student",
        sent_at="2026-03-20T11:00:00",
        caption="My study notes on normalization algorithms.\nNormalization_Summary.pdf (file attached)",
        status="processed",
        db_path=db_path,
    )

    # Add chunks for SQLite FTS5 search
    with db_session(db_path) as conn:
        conn.execute("INSERT INTO chunks (id, file_id, page, text, chunk_index) VALUES ('ca1', ?, 1, 'Database normalization ensures schema integrity and eliminates redundancy.', 0);", (fa,))
        conn.execute("INSERT INTO chunks (id, file_id, page, text, chunk_index) VALUES ('cb1', ?, 1, 'Normalization eliminates update anomalies and maintains functional dependencies.', 0);", (fb,))

    # Attachment refs
    upsert_attachment_ref(
        group_id=gid,
        dedup_key="ref_a",
        status="available",
        original_filename="Normalization_Guide.pdf",
        normalized_filename="normalization_guide.pdf",
        sender="Prof. Alan",
        sent_at="2026-03-01T09:00:00",
        caption="Normalization_Guide.pdf (file attached)",
        context="Study guide for 1NF, 2NF and 3NF normalization.\nNormalization_Guide.pdf (file attached)",
        file_id=fa,
        db_path=db_path,
    )

    upsert_attachment_ref(
        group_id=gid,
        dedup_key="ref_b",
        status="available",
        original_filename="Normalization_Summary.pdf",
        normalized_filename="normalization_summary.pdf",
        sender="Alice Student",
        sent_at="2026-03-20T11:00:00",
        caption="Normalization_Summary.pdf (file attached)",
        context="My study notes on normalization algorithms.\nNormalization_Summary.pdf (file attached)",
        file_id=fb,
        db_path=db_path,
    )

    # 1. Search without filters: both documents returned
    res_unfiltered = search_ranked_documents(
        query="normalization",
        db_path=db_path,
        disable_reranker=True,
    )
    assert len(res_unfiltered.documents) == 2
    assert res_unfiltered.excluded_documents_count == 0
    assert res_unfiltered.documents[0].whatsapp_context is not None

    # 2. Search with sender filter = "Prof. Alan"
    filter_prof = WhatsAppFilterCriteria(senders=["Prof. Alan"])
    res_prof = search_ranked_documents(
        query="normalization",
        db_path=db_path,
        disable_reranker=True,
        whatsapp_filters=filter_prof,
    )
    assert len(res_prof.documents) == 1
    assert res_prof.documents[0].file_id == fa
    assert res_prof.documents[0].rank == 1
    assert res_prof.excluded_documents_count == 1
    assert "Context filtering applied: 1 passed, 1 excluded." in res_prof.diagnostics

    # 3. Search with date filter before 2026-03-10
    filter_date = WhatsAppFilterCriteria(date_to="2026-03-10")
    res_date = search_ranked_documents(
        query="normalization",
        db_path=db_path,
        disable_reranker=True,
        whatsapp_filters=filter_date,
    )
    assert len(res_date.documents) == 1
    assert res_date.documents[0].file_id == fa
