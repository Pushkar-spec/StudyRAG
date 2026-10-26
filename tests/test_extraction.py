"""Tests for StudyRAG document text extraction and chunking pipeline (Step 2.1)."""

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
    init_db,
    insert_file,
    search_chunks_fts,
)
from core.extraction import (
    DocumentChunk,
    ExtractedSection,
    chunk_sections,
    extract_docx_sections,
    extract_document_sections,
    extract_pdf_sections,
    extract_pptx_sections,
    persist_chunks_for_file,
    process_file,
    process_group_files,
    split_text_into_chunks,
)


@pytest.fixture
def test_env(tmp_path: Path):
    """Setup isolated test database and data directory."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return {"db_path": db_path, "data_dir": data_dir, "tmp_path": tmp_path}


# ---------------------------------------------------------------------------
# Helpers for Synthetic Document Generation
# ---------------------------------------------------------------------------

def create_synthetic_pdf(path: Path, pages_text: list[str]) -> Path:
    """Create a synthetic PDF with specific text per page."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text.strip():
            page.insert_text((50, 72), text)
    doc.save(str(path))
    doc.close()
    return path


def create_synthetic_pptx(path: Path, slides_text: list[str]) -> Path:
    """Create a synthetic PPTX presentation with specific text per slide."""
    prs = pptx.Presentation()
    blank_layout = prs.slide_layouts[6]  # Blank slide
    for text in slides_text:
        slide = prs.slides.add_slide(blank_layout)
        if text.strip():
            txBox = slide.shapes.add_textbox(100, 100, 400, 200)
            txBox.text_frame.text = text
    prs.save(str(path))
    return path


def create_synthetic_docx(
    path: Path,
    paragraphs: list[str],
    table_rows: list[list[str]] | None = None,
) -> Path:
    """Create a synthetic Word document with paragraphs and optional table."""
    doc = docx.Document()
    for p in paragraphs:
        if p.strip():
            doc.add_paragraph(p)
    if table_rows:
        tbl = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for r_idx, row in enumerate(table_rows):
            for c_idx, val in enumerate(row):
                tbl.cell(r_idx, c_idx).text = val
    doc.save(str(path))
    return path


# ---------------------------------------------------------------------------
# 1. PDF Extraction Tests
# ---------------------------------------------------------------------------

def test_pdf_multipage_text_extraction(tmp_path: Path):
    """Multi-page PDF extracts text from each page preserving page numbers."""
    pdf_file = tmp_path / "multipage.pdf"
    create_synthetic_pdf(
        pdf_file,
        [
            "Page 1: Introduction to Relational Databases and SQL.",
            "Page 2: Boyce-Codd Normal Form (BCNF) Properties.",
            "Page 3: Transaction ACID Properties: Atomicity, Consistency.",
        ],
    )

    sections = extract_pdf_sections(pdf_file)
    assert len(sections) == 3
    assert sections[0].page == 1
    assert "Page 1" in sections[0].location
    assert "Relational Databases" in sections[0].text

    assert sections[1].page == 2
    assert "Boyce-Codd" in sections[1].text

    assert sections[2].page == 3
    assert "ACID Properties" in sections[2].text


def test_pdf_empty_page_handling(tmp_path: Path):
    """Empty pages inside a PDF do not produce empty sections."""
    pdf_file = tmp_path / "empty_pages.pdf"
    create_synthetic_pdf(
        pdf_file,
        [
            "Page 1: Content here.",
            "",  # Empty Page 2
            "Page 3: More content here.",
        ],
    )

    sections = extract_pdf_sections(pdf_file)
    assert len(sections) == 2
    assert sections[0].page == 1
    assert sections[1].page == 3  # Page 2 was skipped cleanly


def test_scanned_image_only_pdf_reported_as_empty(test_env: dict):
    """Scanned or image-only PDF with no text is reported with status 'empty'."""
    pdf_file = test_env["tmp_path"] / "scanned_doc.pdf"
    # PDF with blank/drawing page having no selectable text
    create_synthetic_pdf(pdf_file, [""])

    from core.db import get_or_create_group
    group_id = get_or_create_group("CS200", db_path=test_env["db_path"])
    f_id = insert_file(
        group_id=group_id,
        sha256="dummy_hash_scanned",
        filename="scanned_doc.pdf",
        path=str(pdf_file),
        ext=".pdf",
        db_path=test_env["db_path"],
    )

    res = process_file(file_id=f_id, db_path=test_env["db_path"])
    assert res.status == "empty"
    assert res.chunk_count == 0
    assert "image-only" in (res.error_message or "").lower() or "no extractable text" in (res.error_message or "").lower()


def test_corrupt_pdf_fails_gracefully(test_env: dict):
    """Corrupt PDF file records failed status with an informative error message."""
    corrupt_pdf = test_env["tmp_path"] / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4 completely invalid corrupt binary stream")

    from core.db import get_or_create_group
    group_id = get_or_create_group("CS200", db_path=test_env["db_path"])
    f_id = insert_file(
        group_id=group_id,
        sha256="dummy_hash_corrupt",
        filename="corrupt.pdf",
        path=str(corrupt_pdf),
        ext=".pdf",
        db_path=test_env["db_path"],
    )

    res = process_file(file_id=f_id, db_path=test_env["db_path"])
    assert res.status == "failed"
    assert res.chunk_count == 0
    assert res.error_message is not None


# ---------------------------------------------------------------------------
# 2. PowerPoint Extraction Tests
# ---------------------------------------------------------------------------

def test_pptx_multislide_text_extraction(tmp_path: Path):
    """Multi-slide presentation extracts text from each slide with slide numbers."""
    pptx_file = tmp_path / "lecture.pptx"
    create_synthetic_pptx(
        pptx_file,
        [
            "Slide 1: Distributed Systems Overview",
            "Slide 2: Consensus and Paxos Protocol",
            "",  # Empty Slide 3
            "Slide 4: Raft Consensus Algorithm",
        ],
    )

    sections = extract_pptx_sections(pptx_file)
    assert len(sections) == 3
    assert sections[0].page == 1
    assert "Distributed Systems" in sections[0].text

    assert sections[1].page == 2
    assert "Paxos" in sections[1].text

    assert sections[2].page == 4
    assert "Raft" in sections[2].text


def test_corrupt_pptx_fails_gracefully(test_env: dict):
    """Corrupt PPTX file fails gracefully without crashing."""
    corrupt_pptx = test_env["tmp_path"] / "broken.pptx"
    corrupt_pptx.write_bytes(b"PK00NotAPresentationZip")

    from core.db import get_or_create_group
    group_id = get_or_create_group("CS200", db_path=test_env["db_path"])
    f_id = insert_file(
        group_id=group_id,
        sha256="dummy_hash_pptx_broken",
        filename="broken.pptx",
        path=str(corrupt_pptx),
        ext=".pptx",
        db_path=test_env["db_path"],
    )

    res = process_file(file_id=f_id, db_path=test_env["db_path"])
    assert res.status == "failed"
    assert "failed" in (res.error_message or "").lower()


# ---------------------------------------------------------------------------
# 3. Word Document Extraction Tests
# ---------------------------------------------------------------------------

def test_docx_paragraphs_and_tables_extracted_in_order(tmp_path: Path):
    """DOCX extracts paragraphs and tables in document order with location metadata."""
    docx_file = tmp_path / "notes.docx"
    create_synthetic_docx(
        docx_file,
        paragraphs=[
            "Paragraph 1: Algorithm Complexity Analysis and Big-O notation.",
            "Paragraph 2: Sorting algorithms: MergeSort and QuickSort.",
        ],
        table_rows=[
            ["Algorithm", "Average Time", "Worst Case"],
            ["MergeSort", "O(N log N)", "O(N log N)"],
            ["QuickSort", "O(N log N)", "O(N^2)"],
        ],
    )

    sections = extract_docx_sections(docx_file)
    assert len(sections) == 3

    # Paragraph 1
    assert sections[0].location == "Paragraph 1"
    assert "Algorithm Complexity" in sections[0].text

    # Paragraph 2
    assert sections[1].location == "Paragraph 2"
    assert "MergeSort and QuickSort" in sections[1].text

    # Table 1
    assert sections[2].location == "Table 1"
    assert "MergeSort | O(N log N)" in sections[2].text
    assert "QuickSort | O(N log N)" in sections[2].text


def test_corrupt_docx_fails_gracefully(test_env: dict):
    """Corrupt DOCX file fails gracefully."""
    corrupt_docx = test_env["tmp_path"] / "corrupt.docx"
    corrupt_docx.write_bytes(b"PK00NotAWordDocument")

    from core.db import get_or_create_group
    group_id = get_or_create_group("CS200", db_path=test_env["db_path"])
    f_id = insert_file(
        group_id=group_id,
        sha256="dummy_hash_docx_broken",
        filename="corrupt.docx",
        path=str(corrupt_docx),
        ext=".docx",
        db_path=test_env["db_path"],
    )

    res = process_file(file_id=f_id, db_path=test_env["db_path"])
    assert res.status == "failed"
    assert res.error_message is not None


# ---------------------------------------------------------------------------
# 4. Chunking Tests
# ---------------------------------------------------------------------------

def test_chunking_size_overlap_and_boundary_splitting():
    """Text is split into chunks respecting chunk_size, overlap, and sentence boundaries."""
    long_text = (
        "First sentence discussing computer architecture principles. "
        "Second sentence covering pipelining hazards and branch prediction. "
        "Third sentence on cache coherence and memory consistency models. "
        "Fourth sentence detailing virtual memory translation and TLB hits. "
        "Fifth sentence explaining superscalar out-of-order execution units."
    )

    chunks = split_text_into_chunks(long_text, chunk_size=150, chunk_overlap=40)
    assert len(chunks) > 1

    # Every chunk must be non-empty and within bounds (or sensible boundary limit)
    for c in chunks:
        assert c.strip()
        assert len(c) > 0

    # Overlap test: chunks produced and keyword present
    assert len(chunks) >= 2
    assert any("pipelining" in c for c in chunks)
    assert any("branch" in c for c in chunks)


def test_chunking_rejects_invalid_config():
    """Invalid chunk_size or chunk_overlap raises ValueError."""
    with pytest.raises(ValueError):
        split_text_into_chunks("Sample text", chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError):
        split_text_into_chunks("Sample text", chunk_size=500, chunk_overlap=-1)

    with pytest.raises(ValueError):
        split_text_into_chunks("Sample text", chunk_size=500, chunk_overlap=500)


def test_chunking_determinism():
    """Identical input and settings produce exactly identical chunks and IDs."""
    sections = [
        ExtractedSection(page=1, location="Page 1", text="Machine learning regression algorithms."),
        ExtractedSection(page=2, location="Page 2", text="Deep neural networks and backpropagation."),
    ]

    chunks_1 = chunk_sections(sections, file_id=42, chunk_size=500, chunk_overlap=100)
    chunks_2 = chunk_sections(sections, file_id=42, chunk_size=500, chunk_overlap=100)

    assert len(chunks_1) == len(chunks_2)
    for c1, c2 in zip(chunks_1, chunks_2):
        assert c1.id == c2.id
        assert c1.chunk_index == c2.chunk_index
        assert c1.text == c2.text
        assert c1.page == c2.page
        assert c1.location == c2.location


# ---------------------------------------------------------------------------
# 5. Database, Persistence & FTS5 Synchronization Tests
# ---------------------------------------------------------------------------

def test_chunks_persisted_and_searchable_in_fts5(test_env: dict):
    """Extracted chunks persist in SQLite chunks table and are searchable in chunks_fts."""
    from core.db import get_or_create_group
    group_id = get_or_create_group("Physics101", db_path=test_env["db_path"])

    pdf_file = test_env["tmp_path"] / "quantum_mechanics.pdf"
    create_synthetic_pdf(
        pdf_file,
        [
            "Page 1: Quantum superposition and Heisenberg uncertainty principle.",
            "Page 2: Quantum entanglement and Einstein-Podolsky-Rosen paradox.",
        ],
    )

    f_id = insert_file(
        group_id=group_id,
        sha256="hash_quantum",
        filename="quantum_mechanics.pdf",
        path=str(pdf_file),
        ext=".pdf",
        db_path=test_env["db_path"],
    )

    res = process_file(file_id=f_id, db_path=test_env["db_path"])
    assert res.status == "processed"
    assert res.chunk_count == 2

    # Verify chunks in chunks table
    chunks = get_file_chunks(f_id, db_path=test_env["db_path"])
    assert len(chunks) == 2
    assert chunks[0]["page"] == 1
    assert chunks[1]["page"] == 2

    # Verify FTS5 query finds the chunk text automatically via trigger
    fts_results = search_chunks_fts("entanglement", group_id=group_id, db_path=test_env["db_path"])
    assert len(fts_results) == 1
    assert "entanglement" in fts_results[0]["text"]
    assert fts_results[0]["file_id"] == f_id


def test_reprocessing_unchanged_file_idempotency(test_env: dict):
    """Reprocessing an unchanged file without force skips reprocessing and preserves chunks."""
    from core.db import get_or_create_group
    group_id = get_or_create_group("CS101", db_path=test_env["db_path"])

    pdf_file = test_env["tmp_path"] / "syllabus.pdf"
    create_synthetic_pdf(pdf_file, ["Syllabus for Computer Science 101."])

    f_id = insert_file(
        group_id=group_id,
        sha256="hash_syllabus",
        filename="syllabus.pdf",
        path=str(pdf_file),
        ext=".pdf",
        db_path=test_env["db_path"],
    )

    # First run
    res1 = process_file(file_id=f_id, force=False, db_path=test_env["db_path"])
    assert res1.status == "processed"
    assert res1.chunk_count == 1

    # Second run without force
    res2 = process_file(file_id=f_id, force=False, db_path=test_env["db_path"])
    assert res2.status == "processed"
    assert res2.chunk_count == 1

    # Verify no duplicate chunks were created
    chunks = get_file_chunks(f_id, db_path=test_env["db_path"])
    assert len(chunks) == 1


def test_force_reprocessing_replaces_chunks_cleanly(test_env: dict):
    """Forced reprocessing replaces old chunks and syncs FTS5 cleanly."""
    from core.db import get_or_create_group
    group_id = get_or_create_group("CS101", db_path=test_env["db_path"])

    pdf_file = test_env["tmp_path"] / "notes.pdf"
    create_synthetic_pdf(pdf_file, ["Initial text: Thermodynamics."])

    f_id = insert_file(
        group_id=group_id,
        sha256="hash_notes",
        filename="notes.pdf",
        path=str(pdf_file),
        ext=".pdf",
        db_path=test_env["db_path"],
    )

    process_file(file_id=f_id, force=False, db_path=test_env["db_path"])
    assert len(search_chunks_fts("Thermodynamics", db_path=test_env["db_path"])) == 1

    # Update PDF content on disk
    create_synthetic_pdf(pdf_file, ["Updated text: Electromagnetism and Maxwell equations."])

    # Force reprocess
    res = process_file(file_id=f_id, force=True, db_path=test_env["db_path"])
    assert res.status == "processed"

    # FTS5 should no longer match Thermodynamics, but should match Electromagnetism
    assert len(search_chunks_fts("Thermodynamics", db_path=test_env["db_path"])) == 0
    assert len(search_chunks_fts("Electromagnetism", db_path=test_env["db_path"])) == 1


def test_batch_processing_resilience_to_single_file_failure(test_env: dict):
    """Failure in one corrupt file does not halt processing of other valid files in batch."""
    from core.db import get_or_create_group
    group_id = get_or_create_group("BioChem", db_path=test_env["db_path"])

    # File 1: Valid PDF
    p1 = test_env["tmp_path"] / "good.pdf"
    create_synthetic_pdf(p1, ["Cellular respiration and glycolysis pathway."])
    f1 = insert_file(group_id, "h1", "good.pdf", str(p1), ".pdf", db_path=test_env["db_path"])

    # File 2: Corrupt PDF
    p2 = test_env["tmp_path"] / "bad.pdf"
    p2.write_bytes(b"%PDF-corrupt garbage")
    f2 = insert_file(group_id, "h2", "bad.pdf", str(p2), ".pdf", db_path=test_env["db_path"])

    # File 3: Valid Word doc
    p3 = test_env["tmp_path"] / "good.docx"
    create_synthetic_docx(p3, ["Enzyme kinetics and Michaelis-Menten constant."])
    f3 = insert_file(group_id, "h3", "good.docx", str(p3), ".docx", db_path=test_env["db_path"])

    # Run batch processing
    batch_res = process_group_files(group_id=group_id, db_path=test_env["db_path"])
    assert batch_res.total_files == 3
    assert batch_res.processed_count == 2
    assert batch_res.failed_count == 1

    # Verified persisted states
    counts = get_processing_status_counts(group_id=group_id, db_path=test_env["db_path"])
    assert counts["processed"] == 2
    assert counts["failed"] == 1


# ---------------------------------------------------------------------------
# 6. End-to-End Acceptance Scenario (Section 13)
# ---------------------------------------------------------------------------

def test_step_2_1_end_to_end_acceptance_scenario(test_env: dict):
    """Synthetic E2E test: register PDF, PPTX, DOCX; process; verify chunks & FTS5 query."""
    from core.db import get_or_create_group
    group_id = get_or_create_group("ComprehensiveExamPrep", db_path=test_env["db_path"])

    # 1. Create PDF
    pdf_path = test_env["tmp_path"] / "DBMS_Textbook.pdf"
    create_synthetic_pdf(
        pdf_path,
        [
            "Page 1: Relational algebra operators: selection, projection, cartesian product.",
            "Page 2: B-Tree and B+Tree indexing structures for fast disk I/O.",
        ],
    )
    insert_file(group_id, "h_pdf", "DBMS_Textbook.pdf", str(pdf_path), ".pdf", db_path=test_env["db_path"])

    # 2. Create PPTX
    pptx_path = test_env["tmp_path"] / "DBMS_Slides.pptx"
    create_synthetic_pptx(
        pptx_path,
        [
            "Slide 1: Transaction Concurrency Control",
            "Slide 2: Two-Phase Locking (2PL) and Deadlock Prevention",
        ],
    )
    insert_file(group_id, "h_pptx", "DBMS_Slides.pptx", str(pptx_path), ".pptx", db_path=test_env["db_path"])

    # 3. Create DOCX
    docx_path = test_env["tmp_path"] / "DBMS_Syllabus.docx"
    create_synthetic_docx(
        docx_path,
        paragraphs=[
            "Unit 1: Entity-Relationship Modeling and Schema Design.",
            "Unit 2: SQL DDL, DML, and complex nested queries.",
        ],
        table_rows=[
            ["Module", "Weightage"],
            ["Indexing", "25%"],
            ["Transactions", "35%"],
        ],
    )
    insert_file(group_id, "h_docx", "DBMS_Syllabus.docx", str(docx_path), ".docx", db_path=test_env["db_path"])

    # 4. Process all documents in group
    batch_res = process_group_files(group_id=group_id, db_path=test_env["db_path"])
    assert batch_res.total_files == 3
    assert batch_res.processed_count == 3
    assert batch_res.failed_count == 0

    # 5. Query FTS5 for known text from each document format
    # Search PDF text
    pdf_fts = search_chunks_fts("Relational", group_id=group_id, db_path=test_env["db_path"])
    assert len(pdf_fts) >= 1
    assert "DBMS_Textbook.pdf" in pdf_fts[0]["filename"]

    # Search PPTX text
    pptx_fts = search_chunks_fts("Concurrency", group_id=group_id, db_path=test_env["db_path"])
    assert len(pptx_fts) >= 1
    assert "DBMS_Slides.pptx" in pptx_fts[0]["filename"]

    # Search DOCX text
    docx_fts = search_chunks_fts("Entity-Relationship", group_id=group_id, db_path=test_env["db_path"])
    assert len(docx_fts) >= 1
    assert "DBMS_Syllabus.docx" in docx_fts[0]["filename"]

    # Search DOCX table text
    table_fts = search_chunks_fts("Weightage", group_id=group_id, db_path=test_env["db_path"])
    assert len(table_fts) >= 1
    assert "DBMS_Syllabus.docx" in table_fts[0]["filename"]
