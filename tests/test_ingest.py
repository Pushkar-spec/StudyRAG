"""Tests for StudyRAG ingestion pipeline (core/ingest.py)."""

import hashlib
import io
from pathlib import Path
import sqlite3
import zipfile
import pytest

from core.db import (
    get_connection,
    get_file_by_sha256,
    get_sender,
    init_db,
    register_sender,
)
from core.ingest import (
    SUPPORTED_EXTENSIONS,
    compute_sha256,
    ingest_folder,
    ingest_zip,
    is_safe_zip_entry,
)


@pytest.fixture
def test_env(tmp_path: Path):
    """Setup isolated test database and data directory."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return {"db_path": db_path, "data_dir": data_dir, "tmp_path": tmp_path}


def create_zip_archive(archive_path: Path, files: dict[str, bytes | str]) -> Path:
    """Helper to create a zip archive with specific file entries."""
    with zipfile.ZipFile(archive_path, "w") as zf:
        for name, content in files.items():
            if isinstance(content, str):
                zf.writestr(name, content.encode("utf-8"))
            else:
                zf.writestr(name, content)
    return archive_path


def test_valid_zip_with_multiple_supported_types(test_env: dict):
    """Verify ingestion of valid ZIP with pdf, pptx, docx, and images."""
    zip_path = test_env["tmp_path"] / "export.zip"
    files = {
        "_chat.txt": (
            "12/03/24, 10:00 am - Messages and calls are end-to-end encrypted.\n"
            "12/03/24, 10:15 am - Prof. Alan: Here are the notes <attached: notes.pdf>\n"
            "12/03/24, 10:16 am - Prof. Alan: slides.pptx (file attached)\n"
            "12/03/24, 10:17 am - Alice: assignment.docx (file attached)\n"
            "12/03/24, 10:18 am - Bob: diagram.png (file attached)\n"
        ),
        "notes.pdf": b"%PDF-1.4 sample pdf content",
        "slides.pptx": b"PK-pptx sample content",
        "assignment.docx": b"PK-docx sample content",
        "diagram.png": b"\x89PNG\r\n\x1a\n sample png",
        "photo.jpg": b"\xff\xd8\xff sample jpg",
        "notes.md": "# Readme markdown (unsupported)",
        "script.py": "print('hello') # unsupported",
    }
    create_zip_archive(zip_path, files)

    result = ingest_zip(
        zip_path=zip_path,
        group_name="CS101",
        db_path=test_env["db_path"],
        data_dir=test_env["data_dir"],
    )

    assert result.imported_count == 5
    assert result.total_found == 5
    assert result.unsupported_count == 2
    assert result.failed_count == 0
    assert len(result.errors) == 0

    # Verify rows in files table
    with get_connection(test_env["db_path"]) as conn:
        cur = conn.cursor()
        cur.execute("SELECT filename, sender, ext FROM files WHERE group_id = ? ORDER BY filename;", (result.group_id,))
        rows = cur.fetchall()
        filenames = [r["filename"] for r in rows]
        assert "notes.pdf" in filenames
        assert "slides.pptx" in filenames
        assert "assignment.docx" in filenames
        assert "diagram.png" in filenames
        assert "photo.jpg" in filenames
        assert "notes.md" not in filenames
        assert "script.py" not in filenames


def test_plain_folder_import_without_chat(test_env: dict):
    """Import documents from folder when no chat log exists."""
    source_dir = test_env["tmp_path"] / "my_docs"
    source_dir.mkdir()
    (source_dir / "lecture1.pdf").write_bytes(b"%PDF-1.4 doc1")
    (source_dir / "lecture2.pdf").write_bytes(b"%PDF-1.4 doc2")
    (source_dir / "random.tmp").write_bytes(b"temp")

    result = ingest_folder(
        folder_path=source_dir,
        group_name="Maths",
        db_path=test_env["db_path"],
        data_dir=test_env["data_dir"],
    )

    assert result.imported_count == 2
    assert result.unmatched_chat_count == 2
    assert result.unsupported_count == 1
    assert result.failed_count == 0

    with get_connection(test_env["db_path"]) as conn:
        cur = conn.cursor()
        cur.execute("SELECT filename, sender, caption FROM files WHERE group_id = ?;", (result.group_id,))
        rows = cur.fetchall()
        for r in rows:
            assert r["sender"] is None
            assert r["caption"] is None


def test_zero_document_zip_warning(test_env: dict):
    """A ZIP containing only text or unsupported files produces a clear warning."""
    zip_path = test_env["tmp_path"] / "text_only.zip"
    files = {
        "_chat.txt": "12/03/24, 10:15 am - Alice: Hello world with no attachments",
        "README.txt": "Just a readme text file",
    }
    create_zip_archive(zip_path, files)

    result = ingest_zip(
        zip_path=zip_path,
        group_name="ChatOnly",
        db_path=test_env["db_path"],
        data_dir=test_env["data_dir"],
    )

    assert result.total_found == 0
    assert result.imported_count == 0
    assert any("zero supported documents" in w.lower() for w in result.warnings)


def test_zip_slip_and_absolute_path_rejection(test_env: dict):
    """Entries with path traversal or absolute paths must be safely rejected."""
    zip_path = test_env["tmp_path"] / "malicious.zip"

    # Create archive with malicious entry names
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../../escaped.pdf", b"evil content")
        zf.writestr("/absolute/evil.pdf", b"evil content")
        zf.writestr("safe_doc.pdf", b"good content")

    result = ingest_zip(
        zip_path=zip_path,
        group_name="SecurityTest",
        db_path=test_env["db_path"],
        data_dir=test_env["data_dir"],
    )

    # safe_doc.pdf should be imported, the two traversal entries rejected
    assert result.imported_count == 1
    assert result.failed_count == 2
    assert any("ZIP Slip" in w for w in result.warnings)
    assert not (test_env["tmp_path"] / "escaped.pdf").exists()


def test_corrupt_zip_handling(test_env: dict):
    """Corrupt or non-ZIP file produces a clean error result without crashing."""
    corrupt_file = test_env["tmp_path"] / "corrupt.zip"
    corrupt_file.write_bytes(b"This is definitely not a valid zip archive file data")

    result = ingest_zip(
        zip_path=corrupt_file,
        group_name="CorruptGroup",
        db_path=test_env["db_path"],
        data_dir=test_env["data_dir"],
    )

    assert result.imported_count == 0
    assert len(result.errors) > 0
    assert "not a valid ZIP" in result.errors[0]


def test_duplicate_file_content_with_different_filenames(test_env: dict):
    """Files with identical SHA-256 within the same group are deduplicated."""
    zip_path = test_env["tmp_path"] / "duplicates.zip"
    shared_content = b"%PDF-1.4 identical content for both files"
    files = {
        "original.pdf": shared_content,
        "copy_of_original.pdf": shared_content,
    }
    create_zip_archive(zip_path, files)

    result = ingest_zip(
        zip_path=zip_path,
        group_name="DeduplicationGroup",
        db_path=test_env["db_path"],
        data_dir=test_env["data_dir"],
    )

    # 1 imported, 1 skipped as duplicate
    assert result.total_found == 2
    assert result.imported_count == 1
    assert result.duplicates_skipped == 1


def test_reimporting_same_zip_is_idempotent(test_env: dict):
    """Reimporting the exact same ZIP must not create duplicate rows or errors."""
    zip_path = test_env["tmp_path"] / "idempotent.zip"
    files = {
        "syllabus.pdf": b"%PDF-1.4 syllabus",
        "homework.docx": b"PK-homework docx",
    }
    create_zip_archive(zip_path, files)

    # First run
    r1 = ingest_zip(
        zip_path=zip_path,
        group_name="IdempotencyTest",
        db_path=test_env["db_path"],
        data_dir=test_env["data_dir"],
    )
    assert r1.imported_count == 2
    assert r1.duplicates_skipped == 0

    # Second run
    r2 = ingest_zip(
        zip_path=zip_path,
        group_name="IdempotencyTest",
        db_path=test_env["db_path"],
        data_dir=test_env["data_dir"],
    )
    assert r2.imported_count == 0
    assert r2.duplicates_skipped == 2
    assert r2.failed_count == 0

    # Check total rows in files table for this group is still 2
    with get_connection(test_env["db_path"]) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM files WHERE group_id = ?;", (r1.group_id,))
        assert cur.fetchone()[0] == 2


def test_duplicate_filenames_with_different_content(test_env: dict):
    """Same filename with different content is disambiguated safely on extraction."""
    # First archive has notes.pdf with content A
    zip1 = test_env["tmp_path"] / "export1.zip"
    create_zip_archive(zip1, {"notes.pdf": b"%PDF-1.4 Content A"})
    r1 = ingest_zip(zip1, "CollisionTest", db_path=test_env["db_path"], data_dir=test_env["data_dir"])
    assert r1.imported_count == 1

    # Second archive has notes.pdf with content B
    zip2 = test_env["tmp_path"] / "export2.zip"
    create_zip_archive(zip2, {"notes.pdf": b"%PDF-1.4 Content B - Different"})
    r2 = ingest_zip(zip2, "CollisionTest", db_path=test_env["db_path"], data_dir=test_env["data_dir"])
    assert r2.imported_count == 1
    assert r2.duplicates_skipped == 0

    # Both distinct files exist in DB
    with get_connection(test_env["db_path"]) as conn:
        cur = conn.cursor()
        cur.execute("SELECT filename, sha256 FROM files WHERE group_id = ?;", (r1.group_id,))
        rows = cur.fetchall()
        assert len(rows) == 2


def test_source_files_remain_unchanged(test_env: dict):
    """Source files (ZIP and folder) are never modified or deleted."""
    source_folder = test_env["tmp_path"] / "source"
    source_folder.mkdir()
    doc_path = source_folder / "test.pdf"
    doc_path.write_bytes(b"%PDF original content")
    hash_before = compute_sha256(doc_path)

    ingest_folder(
        folder_path=source_folder,
        group_name="PreserveSource",
        db_path=test_env["db_path"],
        data_dir=test_env["data_dir"],
    )

    assert doc_path.exists()
    assert compute_sha256(doc_path) == hash_before


def test_message_linking_surrounding_window_and_sender(test_env: dict):
    """Verify surrounding messages from same sender within 5 minutes are captured."""
    zip_path = test_env["tmp_path"] / "chat_link.zip"
    chat_text = (
        "12/03/24, 10:00 am - Prof. Alan: Message 15 minutes before (should be excluded)\n"
        "12/03/24, 10:12 am - Prof. Alan: Important context 3 mins before attachment\n"
        "12/03/24, 10:14 am - Bob: Student question (different sender, should be excluded)\n"
        "12/03/24, 10:15 am - Prof. Alan: Here is Chapter 4 <attached: chapter4.pdf>\n"
        "12/03/24, 10:17 am - Prof. Alan: Please read pages 10-20 (2 mins after)\n"
        "12/03/24, 10:18 am - Prof. Alan: Exam has 3 questions on this (3 mins after)\n"
        "12/03/24, 10:19 am - Prof. Alan: Message 4 after (limit is 3 surrounding messages)\n"
        "12/03/24, 10:25 am - Prof. Alan: Message 10 minutes after (outside window)\n"
    )
    files = {
        "_chat.txt": chat_text,
        "chapter4.pdf": b"%PDF-1.4 chapter 4 content",
    }
    create_zip_archive(zip_path, files)

    result = ingest_zip(
        zip_path=zip_path,
        group_name="Physics",
        db_path=test_env["db_path"],
        data_dir=test_env["data_dir"],
    )

    assert result.imported_count == 1

    with get_connection(test_env["db_path"]) as conn:
        cur = conn.cursor()
        cur.execute("SELECT sender, sent_at, caption FROM files WHERE group_id = ?;", (result.group_id,))
        row = cur.fetchone()

        assert row["sender"] == "Prof. Alan"
        assert row["sent_at"] == "2024-03-12T10:15:00"

        caption = row["caption"]
        assert caption is not None
        # Included messages:
        assert "Important context 3 mins before attachment" in caption
        assert "Here is Chapter 4" in caption
        assert "Please read pages 10-20" in caption
        assert "Exam has 3 questions on this" in caption

        # Excluded messages:
        assert "Message 15 minutes before" not in caption  # Outside window before
        assert "Student question" not in caption  # Different sender
        assert "Message 10 minutes after" not in caption  # Outside window after


def test_missing_attachment_produces_warning(test_env: dict):
    """When chat references a file that is not in the archive, a warning is recorded."""
    zip_path = test_env["tmp_path"] / "missing_att.zip"
    files = {
        "_chat.txt": "12/03/24, 10:15 am - Alice: <attached: missing_doc.pdf>",
        "actual_doc.pdf": b"%PDF actual",
    }
    create_zip_archive(zip_path, files)

    result = ingest_zip(
        zip_path=zip_path,
        group_name="MissingAttachmentGroup",
        db_path=test_env["db_path"],
        data_dir=test_env["data_dir"],
    )

    assert result.imported_count == 1
    assert any("missing_doc.pdf" in w for w in result.warnings)


def test_professor_flag_preserved_on_reimport(test_env: dict):
    """Existing is_professor status on a sender must not be reset to 0 on re-ingestion."""
    zip_path = test_env["tmp_path"] / "prof.zip"
    files = {
        "_chat.txt": "12/03/24, 10:15 am - Prof. Turing: Lecture notes <attached: notes.pdf>",
        "notes.pdf": b"%PDF turing notes",
    }
    create_zip_archive(zip_path, files)

    # First import
    r1 = ingest_zip(zip_path, "CS", db_path=test_env["db_path"], data_dir=test_env["data_dir"])

    # Manually promote Prof. Turing to professor (is_professor = 1)
    with get_connection(test_env["db_path"]) as conn:
        conn.execute(
            "UPDATE senders SET is_professor = 1 WHERE group_id = ? AND name = 'Prof. Turing';",
            (r1.group_id,),
        )
        conn.commit()

    sender_before = get_sender(r1.group_id, "Prof. Turing", db_path=test_env["db_path"])
    assert sender_before["is_professor"] == 1

    # Reimport the group
    ingest_zip(zip_path, "CS", db_path=test_env["db_path"], data_dir=test_env["data_dir"])

    # Verify is_professor is still 1
    sender_after = get_sender(r1.group_id, "Prof. Turing", db_path=test_env["db_path"])
    assert sender_after["is_professor"] == 1


def test_ambiguous_duplicate_attachment_filenames(test_env: dict):
    """When multiple chat messages reference the same attachment name, avoid wrong caption assignment."""
    zip_path = test_env["tmp_path"] / "ambiguous.zip"
    files = {
        "_chat.txt": (
            "12/03/24, 10:15 am - Alice: <attached: notes.pdf> Alice's notes\n"
            "12/03/24, 11:30 am - Bob: <attached: notes.pdf> Bob's notes\n"
        ),
        "notes.pdf": b"%PDF-1.4 sample",
    }
    create_zip_archive(zip_path, files)

    result = ingest_zip(
        zip_path=zip_path,
        group_name="AmbiguousGroup",
        db_path=test_env["db_path"],
        data_dir=test_env["data_dir"],
    )

    assert result.imported_count == 1
    assert any("Ambiguous attachment" in w for w in result.warnings)

    with get_connection(test_env["db_path"]) as conn:
        cur = conn.cursor()
        cur.execute("SELECT sender, caption FROM files WHERE group_id = ?;", (result.group_id,))
        row = cur.fetchone()
        # Skipped automated assignment to avoid assigning wrong sender/caption
        assert row["sender"] is None
        assert row["caption"] is None
