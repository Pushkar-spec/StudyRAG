"""Tests for StudyRAG persistent missing-attachment tracking and recovery (Step 1.3)."""

from pathlib import Path
import sqlite3
import zipfile
import pytest

from core.chat_parser import parse_chat
from core.db import (
    get_all_groups,
    get_attachment_ref,
    get_attachment_ref_counts,
    get_attachment_refs,
    get_connection,
    get_file_by_id,
    get_group_files,
    init_db,
)
from core.ingest import (
    RecoveryResult,
    compute_sha256,
    get_candidates_for_ref,
    ingest_folder,
    ingest_zip,
    link_attachment_ref_manually,
    reconcile_attachment_refs,
    recover_file_or_folder,
)


@pytest.fixture
def test_env(tmp_path: Path):
    """Setup isolated test database and data directory."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return {"db_path": db_path, "data_dir": data_dir, "tmp_path": tmp_path}


def create_zip(archive_path: Path, files: dict[str, bytes | str]) -> Path:
    """Helper to create a zip archive."""
    with zipfile.ZipFile(archive_path, "w") as zf:
        for name, content in files.items():
            if isinstance(content, str):
                zf.writestr(name, content.encode("utf-8"))
            else:
                zf.writestr(name, content)
    return archive_path


# ---------------------------------------------------------------------------
# 1. Detection Tests
# ---------------------------------------------------------------------------

def test_attachment_present_recorded_as_available(test_env: dict):
    """An explicitly referenced attachment that exists in the export is recorded as available."""
    zip_p = test_env["tmp_path"] / "present.zip"
    files = {
        "_chat.txt": "12/03/24, 10:15 am - Prof. Alan: <attached: syllabus.pdf> Course overview\n",
        "syllabus.pdf": b"%PDF-1.4 sample syllabus content",
    }
    create_zip(zip_p, files)

    res = ingest_zip(zip_p, "CourseA", db_path=test_env["db_path"], data_dir=test_env["data_dir"])
    assert res.imported_count == 1

    refs = get_attachment_refs(group_id=res.group_id, db_path=test_env["db_path"])
    assert len(refs) == 1
    ref = refs[0]
    assert ref["status"] == "available"
    assert ref["original_filename"] == "syllabus.pdf"
    assert ref["sender"] == "Prof. Alan"
    assert ref["file_id"] is not None
    assert ref["resolution_method"] == "automatic"


def test_attachment_absent_recorded_as_missing(test_env: dict):
    """An attachment referenced in chat but missing from export is persisted as missing."""
    zip_p = test_env["tmp_path"] / "missing.zip"
    files = {
        "_chat.txt": "12/03/24, 10:15 am - Prof. Alan: <attached: DBMS_Unit_5.pdf> Study for quiz\n",
    }
    create_zip(zip_p, files)

    res = ingest_zip(zip_p, "DBMS", db_path=test_env["db_path"], data_dir=test_env["data_dir"])
    assert res.imported_count == 0
    assert res.missing_attachments_count == 1

    refs = get_attachment_refs(group_id=res.group_id, status="missing", db_path=test_env["db_path"])
    assert len(refs) == 1
    ref = refs[0]
    assert ref["original_filename"] == "DBMS_Unit_5.pdf"
    assert ref["sender"] == "Prof. Alan"
    assert ref["sent_at"] == "2024-03-12T10:15:00"
    assert "Study for quiz" in ref["caption"]
    assert ref["file_id"] is None
    assert ref["status"] == "missing"


def test_text_only_export_preserves_context(test_env: dict):
    """A text-only export with missing attachments retains sender, timestamp, and message context."""
    zip_p = test_env["tmp_path"] / "text_only.zip"
    files = {
        "_chat.txt": (
            "12/03/24, 10:10 am - Dr. Stone: Exam instructions\n"
            "12/03/24, 10:12 am - Dr. Stone: Review formulas carefully\n"
            "12/03/24, 10:15 am - Dr. Stone: <attached: Formula_Sheet.pdf> Final formulas\n"
            "12/03/24, 10:16 am - Dr. Stone: Bring this sheet to exam hall\n"
        )
    }
    create_zip(zip_p, files)

    res = ingest_zip(zip_p, "Chemistry", db_path=test_env["db_path"], data_dir=test_env["data_dir"])
    assert res.total_found == 0
    assert any("zero supported documents" in w.lower() for w in res.warnings)

    counts = get_attachment_ref_counts(group_id=res.group_id, db_path=test_env["db_path"])
    assert counts["missing"] == 1

    refs = get_attachment_refs(group_id=res.group_id, db_path=test_env["db_path"])
    ref = refs[0]
    assert ref["original_filename"] == "Formula_Sheet.pdf"
    assert ref["sender"] == "Dr. Stone"
    assert "Review formulas carefully" in ref["context"]
    assert "Bring this sheet to exam hall" in ref["context"]


def test_attachment_without_filename_retained_without_inventing_one(test_env: dict):
    """Omitted media references without explicit filenames are retained with message context."""
    zip_p = test_env["tmp_path"] / "omitted.zip"
    files = {
        "_chat.txt": "12/03/24, 10:15 am - Alice: <Media omitted> Important notes\n",
    }
    create_zip(zip_p, files)

    res = ingest_zip(zip_p, "GroupOmitted", db_path=test_env["db_path"], data_dir=test_env["data_dir"])
    refs = get_attachment_refs(group_id=res.group_id, db_path=test_env["db_path"])
    assert len(refs) == 1
    ref = refs[0]
    assert ref["original_filename"] is None
    assert ref["normalized_filename"] is None
    assert ref["status"] == "missing"
    assert "Important notes" in ref["caption"]


def test_ordinary_text_does_not_create_false_missing_attachment(test_env: dict):
    """Ordinary conversation mentioning 'PDF' or 'notes' does not create an attachment reference."""
    zip_p = test_env["tmp_path"] / "chat_talk.zip"
    files = {
        "_chat.txt": (
            "12/03/24, 10:15 am - Alice: Did anyone read the PDF notes for Unit 4?\n"
            "12/03/24, 10:16 am - Bob: Yes, the slides and document are great.\n"
        )
    }
    create_zip(zip_p, files)

    res = ingest_zip(zip_p, "ChatTalk", db_path=test_env["db_path"], data_dir=test_env["data_dir"])
    refs = get_attachment_refs(group_id=res.group_id, db_path=test_env["db_path"])
    assert len(refs) == 0
    assert res.missing_attachments_count == 0


def test_reimporting_zip_does_not_duplicate_references(test_env: dict):
    """Reimporting the same ZIP multiple times must not duplicate attachment references."""
    zip_p = test_env["tmp_path"] / "reimport_test.zip"
    files = {
        "_chat.txt": "12/03/24, 10:15 am - Prof. Alan: <attached: missing_doc.pdf> Notes\n",
    }
    create_zip(zip_p, files)

    r1 = ingest_zip(zip_p, "ReimportGroup", db_path=test_env["db_path"], data_dir=test_env["data_dir"])
    r2 = ingest_zip(zip_p, "ReimportGroup", db_path=test_env["db_path"], data_dir=test_env["data_dir"])

    refs = get_attachment_refs(group_id=r1.group_id, db_path=test_env["db_path"])
    assert len(refs) == 1


# ---------------------------------------------------------------------------
# 2. Matching Tests
# ---------------------------------------------------------------------------

def test_safe_normalized_filename_matching(test_env: dict):
    """Normalized filename matching handles case differences and whitespace safely."""
    # Chat references "Lecture_01.PDF"
    zip_p = test_env["tmp_path"] / "norm_test.zip"
    files = {
        "_chat.txt": "12/03/24, 10:15 am - Alice: <attached: Lecture_01.PDF> First lecture\n",
        "lecture_01.pdf": b"%PDF-1.4 lecture content",
    }
    create_zip(zip_p, files)

    res = ingest_zip(zip_p, "NormGroup", db_path=test_env["db_path"], data_dir=test_env["data_dir"])
    assert res.imported_count == 1

    refs = get_attachment_refs(group_id=res.group_id, status="available", db_path=test_env["db_path"])
    assert len(refs) == 1
    assert refs[0]["resolution_method"] == "automatic"


def test_multiple_candidates_produce_ambiguous_status(test_env: dict):
    """When multiple candidate files or references exist, status must be ambiguous."""
    zip_p = test_env["tmp_path"] / "ambig.zip"
    # Chat has two distinct messages referencing "notes.pdf"
    files = {
        "_chat.txt": (
            "12/03/24, 10:00 am - Alice: <attached: notes.pdf> Alice version\n"
            "12/03/24, 11:00 am - Bob: <attached: notes.pdf> Bob version\n"
        ),
        "notes.pdf": b"%PDF-1.4 notes content",
    }
    create_zip(zip_p, files)

    res = ingest_zip(zip_p, "AmbigTest", db_path=test_env["db_path"], data_dir=test_env["data_dir"])
    counts = get_attachment_ref_counts(group_id=res.group_id, db_path=test_env["db_path"])
    assert counts["ambiguous"] == 2
    assert counts["available"] == 0


def test_cross_group_matches_disallowed_automatically(test_env: dict):
    """A file in Group B cannot automatically resolve a missing reference in Group A."""
    # Group A: missing "notes.pdf"
    zip_a = test_env["tmp_path"] / "group_a.zip"
    create_zip(zip_a, {"_chat.txt": "12/03/24, 10:15 am - Prof: <attached: notes.pdf> Group A notes\n"})
    res_a = ingest_zip(zip_a, "GroupA", db_path=test_env["db_path"], data_dir=test_env["data_dir"])

    # Group B: has "notes.pdf"
    zip_b = test_env["tmp_path"] / "group_b.zip"
    create_zip(zip_b, {"notes.pdf": b"%PDF-1.4 Group B notes"})
    res_b = ingest_zip(zip_b, "GroupB", db_path=test_env["db_path"], data_dir=test_env["data_dir"])

    # Group A reference must still be missing
    refs_a = get_attachment_refs(group_id=res_a.group_id, db_path=test_env["db_path"])
    assert refs_a[0]["status"] == "missing"


# ---------------------------------------------------------------------------
# 3. Recovery Workflow Tests
# ---------------------------------------------------------------------------

def test_later_file_recovery_workflow(test_env: dict):
    """Importing a missing file later via recovery resolves the reference unambiguously."""
    # Step 1: Export missing DBMS_Unit_5.pdf
    zip_p = test_env["tmp_path"] / "initial_export.zip"
    files = {
        "_chat.txt": "12/03/24, 10:15 am - Prof. Alan: <attached: DBMS_Unit_5.pdf> Read for midterm\n",
    }
    create_zip(zip_p, files)

    r_init = ingest_zip(zip_p, "DBMS_Class", db_path=test_env["db_path"], data_dir=test_env["data_dir"])
    assert r_init.missing_attachments_count == 1

    refs_before = get_attachment_refs(group_id=r_init.group_id, status="missing", db_path=test_env["db_path"])
    assert len(refs_before) == 1
    ref_id = refs_before[0]["id"]

    # Step 2: Student downloads DBMS_Unit_5.pdf later
    downloaded_pdf = test_env["tmp_path"] / "DBMS_Unit_5.pdf"
    downloaded_pdf.write_bytes(b"%PDF-1.4 DBMS Unit 5 Full Chapter Content")

    # Step 3: Run recovery workflow
    rec_res = recover_file_or_folder(
        target_path=downloaded_pdf,
        group_id=r_init.group_id,
        db_path=test_env["db_path"],
        data_dir=test_env["data_dir"],
    )

    assert isinstance(rec_res, RecoveryResult)
    assert len(rec_res.imported_files) == 1
    assert ref_id in rec_res.resolved_references
    assert rec_res.remaining_missing == 0

    # Step 4: Verify reference is now available and points to registered file
    ref_after = get_attachment_ref(ref_id, db_path=test_env["db_path"])
    assert ref_after["status"] == "available"
    assert ref_after["resolution_method"] == "automatic"
    assert ref_after["file_id"] is not None

    # Step 5: Verify file in DB has inherited the chat context
    file_row = get_file_by_id(ref_after["file_id"], db_path=test_env["db_path"])
    assert file_row["sender"] == "Prof. Alan"
    assert "Read for midterm" in file_row["caption"]

    # Step 6: Idempotent re-recovery
    rec_res2 = recover_file_or_folder(
        target_path=downloaded_pdf,
        group_id=r_init.group_id,
        db_path=test_env["db_path"],
        data_dir=test_env["data_dir"],
    )
    assert rec_res2.duplicates_skipped == 1
    assert rec_res2.remaining_missing == 0


def test_manual_linking_workflow(test_env: dict):
    """User can manually link an ambiguous or missing reference to a registered file."""
    # Chat references "unit_notes.pdf"
    zip_p = test_env["tmp_path"] / "manual_test.zip"
    files = {
        "_chat.txt": "12/03/24, 10:15 am - Prof. Alan: <attached: unit_notes.pdf> Official notes\n",
        "lecture_notes.pdf": b"%PDF-1.4 file with slightly different name",
    }
    create_zip(zip_p, files)

    res = ingest_zip(zip_p, "ManualLinkGroup", db_path=test_env["db_path"], data_dir=test_env["data_dir"])
    refs = get_attachment_refs(group_id=res.group_id, status="missing", db_path=test_env["db_path"])
    assert len(refs) == 1
    ref_id = refs[0]["id"]

    group_files = get_group_files(group_id=res.group_id, db_path=test_env["db_path"])
    assert len(group_files) == 1
    target_file_id = group_files[0]["id"]

    # Candidate retrieval
    candidates = get_candidates_for_ref(ref_id, db_path=test_env["db_path"])
    assert len(candidates) >= 1

    # Perform manual link
    ok, msg = link_attachment_ref_manually(ref_id=ref_id, file_id=target_file_id, db_path=test_env["db_path"])
    assert ok is True

    # Verify persisted status and method
    ref_updated = get_attachment_ref(ref_id, db_path=test_env["db_path"])
    assert ref_updated["status"] == "available"
    assert ref_updated["resolution_method"] == "manual"
    assert ref_updated["file_id"] == target_file_id

    # Re-ingesting or re-initializing must not destroy manual resolution
    init_db(test_env["db_path"])
    ref_after_init = get_attachment_ref(ref_id, db_path=test_env["db_path"])
    assert ref_after_init["resolution_method"] == "manual"
    assert ref_after_init["status"] == "available"


# ---------------------------------------------------------------------------
# 4. Section 9 End-to-End Acceptance Scenario
# ---------------------------------------------------------------------------

def test_section_9_end_to_end_acceptance_scenario(test_env: dict):
    """Full execution of the Section 9 acceptance scenario:

    1. Import ZIP with DBMS_Unit_4.pdf (present) and DBMS_Unit_5.pdf (missing).
    2. Verify DBMS_Unit_4.pdf registered & available.
    3. Verify DBMS_Unit_5.pdf recorded as missing with message context.
    4. Simulate downloading DBMS_Unit_5.pdf later and recover it.
    5. Verify DBMS_Unit_5.pdf becomes available and inherits context.
    6. Re-import and verify idempotency.
    7. Ambiguous candidate case requires user review.
    """
    zip_path = test_env["tmp_path"] / "whatsapp_export.zip"
    chat_content = (
        "12/03/24, 09:00 am - Prof. Alan: <attached: DBMS_Unit_4.pdf> Unit 4 Relational Algebra notes\n"
        "12/03/24, 09:05 am - Prof. Alan: <attached: DBMS_Unit_5.pdf> Unit 5 Normalization notes (study for quiz!)\n"
    )
    files = {
        "_chat.txt": chat_content,
        "DBMS_Unit_4.pdf": b"%PDF-1.4 DBMS Unit 4 content",
    }
    create_zip(zip_path, files)

    # 1. Import ZIP
    res = ingest_zip(zip_path, "DBMS_Full", db_path=test_env["db_path"], data_dir=test_env["data_dir"])
    assert res.imported_count == 1

    # 2. Verify DBMS_Unit_4.pdf is available
    refs = get_attachment_refs(group_id=res.group_id, db_path=test_env["db_path"])
    ref_map = {r["original_filename"]: r for r in refs}

    assert "DBMS_Unit_4.pdf" in ref_map
    assert ref_map["DBMS_Unit_4.pdf"]["status"] == "available"
    assert ref_map["DBMS_Unit_4.pdf"]["file_id"] is not None

    # 3. Verify DBMS_Unit_5.pdf is missing
    assert "DBMS_Unit_5.pdf" in ref_map
    ref_u5 = ref_map["DBMS_Unit_5.pdf"]
    assert ref_u5["status"] == "missing"
    assert ref_u5["file_id"] is None
    assert "Unit 5 Normalization notes" in ref_u5["caption"]

    # 4 & 5. Simulate student downloading DBMS_Unit_5.pdf later and running recovery
    later_download = test_env["tmp_path"] / "DBMS_Unit_5.pdf"
    later_download.write_bytes(b"%PDF-1.4 DBMS Unit 5 Normalization Complete")

    rec_result = recover_file_or_folder(
        target_path=later_download,
        group_id=res.group_id,
        db_path=test_env["db_path"],
        data_dir=test_env["data_dir"],
    )
    assert len(rec_result.imported_files) == 1
    assert rec_result.remaining_missing == 0

    # Verify DBMS_Unit_5.pdf is now available
    ref_u5_after = get_attachment_ref(ref_u5["id"], db_path=test_env["db_path"])
    assert ref_u5_after["status"] == "available"
    assert ref_u5_after["resolution_method"] == "automatic"
    assert ref_u5_after["file_id"] is not None

    # 6. Re-import the downloaded file - verify idempotency
    rec_result2 = recover_file_or_folder(
        target_path=later_download,
        group_id=res.group_id,
        db_path=test_env["db_path"],
        data_dir=test_env["data_dir"],
    )
    assert rec_result2.duplicates_skipped == 1

    # 7. Ambiguous case: 2 plausible files exist
    # Suppose another reference for "Practice_Questions.pdf" arrives
    zip_ambig = test_env["tmp_path"] / "ambig_case.zip"
    create_zip(
        zip_ambig,
        {
            "_chat.txt": (
                "12/03/24, 11:00 am - Prof. Alan: <attached: Practice.pdf> Set 1\n"
                "12/03/24, 11:05 am - Prof. Alan: <attached: Practice.pdf> Set 2\n"
            ),
            "Practice.pdf": b"%PDF-1.4 Practice set",
        },
    )
    res_ambig = ingest_zip(zip_ambig, "AmbigCheck", db_path=test_env["db_path"], data_dir=test_env["data_dir"])
    ambig_refs = get_attachment_refs(group_id=res_ambig.group_id, status="ambiguous", db_path=test_env["db_path"])
    # System leaves them ambiguous rather than guessing
    assert len(ambig_refs) == 2


def test_recovery_result_dataclass_contract_and_exports():
    """Verify RecoveryResult contract, default fields, and module export."""
    from core import ingest
    assert hasattr(ingest, "RecoveryResult")
    assert "RecoveryResult" in ingest.__all__

    res = RecoveryResult(group_id=1, group_name="MathGroup")
    assert res.group_id == 1
    assert res.group_name == "MathGroup"
    assert res.imported_files == []
    assert res.duplicates_skipped == 0
    assert res.resolved_references == []
    assert res.remaining_missing == 0
    assert res.ambiguous_references == 0
    assert res.warnings == []
    assert res.errors == []
