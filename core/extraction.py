"""Document text extraction and chunking pipeline for StudyRAG.

Handles PDF (PyMuPDF), PowerPoint (python-pptx), and Word (python-docx) extraction,
preserving page/slide/Word location metadata, deterministic source-aware chunking,
incremental processing, and atomic SQLite chunk persistence with FTS5 synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any

import docx
try:
    import pymupdf as fitz  # PyMuPDF modern import
except ImportError:
    import fitz
import pptx
from pptx.enum.shapes import MSO_SHAPE_TYPE

from core.config import Settings, load
from core.db import (
    db_session,
    get_file_by_id,
    get_file_chunks,
    update_file_processing_status,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ExtractedSection",
    "DocumentChunk",
    "FileProcessingResult",
    "EXTRACTION_SUPPORTED_EXTENSIONS",
    "extract_pdf_sections",
    "extract_pptx_sections",
    "extract_docx_sections",
    "extract_document_sections",
    "split_text_into_chunks",
    "chunk_sections",
    "persist_chunks_for_file",
    "process_file",
    "process_group_files",
    "get_file_chunks",
]

EXTRACTION_SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".txt"}


@dataclass
class ExtractedSection:
    """A bounded section of extracted document text with location metadata."""

    page: int | None
    location: str
    text: str


@dataclass
class DocumentChunk:
    """A processed text chunk mapped to its source document and location."""

    id: str
    file_id: int
    page: int | None
    location: str
    chunk_index: int
    text: str


@dataclass
class FileProcessingResult:
    """Result of processing a single registered document."""

    file_id: int
    filename: str
    status: str  # 'processed', 'empty', 'unsupported', 'failed'
    chunk_count: int = 0
    error_message: str | None = None


@dataclass
class ExtractionBatchResult:
    """Aggregated results of batch document processing."""

    total_files: int = 0
    processed_count: int = 0
    empty_count: int = 0
    unsupported_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    results: list[FileProcessingResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Document Text Extraction
# ---------------------------------------------------------------------------

def extract_pdf_sections(file_path: Path | str) -> list[ExtractedSection]:
    """Extract text from a PDF page by page using PyMuPDF.

    Args:
        file_path: Path to the target PDF file.

    Returns:
        List of ExtractedSection instances, one per non-empty page.

    Raises:
        ValueError: If PDF is encrypted and cannot be opened with empty password.
        Exception: If the PDF is malformed or unreadable.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    doc = fitz.open(str(path))
    try:
        if doc.is_encrypted:
            authenticated = doc.authenticate("")
            if not authenticated:
                raise ValueError("Encrypted PDF: password protected and cannot be extracted")

        sections: list[ExtractedSection] = []
        for page_idx, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                sections.append(
                    ExtractedSection(
                        page=page_idx,
                        location=f"Page {page_idx}",
                        text=text,
                    )
                )
        return sections
    finally:
        doc.close()


def _extract_shape_text(shape: Any) -> list[str]:
    """Recursively extract text strings from a PowerPoint shape, table, or group."""
    texts: list[str] = []
    if shape.has_text_frame:
        for para in shape.text_frame.paragraphs:
            p_text = "".join(run.text for run in para.runs).strip() or para.text.strip()
            if p_text:
                texts.append(p_text)
    elif shape.has_table:
        for row in shape.table.rows:
            row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_cells:
                texts.append(" | ".join(row_cells))
    elif shape.shape_type == MSO_SHAPE_TYPE.GROUP or hasattr(shape, "shapes"):
        for child in shape.shapes:
            texts.extend(_extract_shape_text(child))
    return texts


def extract_pptx_sections(file_path: Path | str) -> list[ExtractedSection]:
    """Extract text from a PowerPoint presentation slide by slide.

    Args:
        file_path: Path to the target PPTX file.

    Returns:
        List of ExtractedSection instances, one per non-empty slide.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PowerPoint file not found: {path}")

    prs = pptx.Presentation(str(path))
    sections: list[ExtractedSection] = []

    for slide_idx, slide in enumerate(prs.slides, start=1):
        slide_texts: list[str] = []
        for shape in slide.shapes:
            slide_texts.extend(_extract_shape_text(shape))

        full_slide_text = "\n\n".join(slide_texts).strip()
        if full_slide_text:
            sections.append(
                ExtractedSection(
                    page=slide_idx,
                    location=f"Slide {slide_idx}",
                    text=full_slide_text,
                )
            )

    return sections


def extract_docx_sections(file_path: Path | str) -> list[ExtractedSection]:
    """Extract text from a Word document preserving paragraph and table order.

    Args:
        file_path: Path to the target DOCX file.

    Returns:
        List of ExtractedSection instances with paragraph and table location metadata.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Word document not found: {path}")

    doc = docx.Document(str(path))
    sections: list[ExtractedSection] = []
    para_num = 0
    table_num = 0

    # Iterate through child elements in document body order
    for child in doc.element.body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "p":
            para = docx.text.paragraph.Paragraph(child, doc)
            text = para.text.strip()
            if text:
                para_num += 1
                sections.append(
                    ExtractedSection(
                        page=para_num,
                        location=f"Paragraph {para_num}",
                        text=text,
                    )
                )
        elif tag == "tbl":
            table_num += 1
            tbl = docx.table.Table(child, doc)
            row_lines: list[str] = []
            for row in tbl.rows:
                # Deduplicate merged cells in row by tracking element identity
                seen_cells: set[Any] = set()
                cell_texts: list[str] = []
                for cell in row.cells:
                    if cell._tc not in seen_cells:
                        seen_cells.add(cell._tc)
                        c_text = cell.text.strip()
                        if c_text:
                            cell_texts.append(c_text)
                if cell_texts:
                    row_lines.append(" | ".join(cell_texts))
            if row_lines:
                sections.append(
                    ExtractedSection(
                        page=None,
                        location=f"Table {table_num}",
                        text="\n".join(row_lines),
                    )
                )

    return sections


def extract_txt_sections(file_path: Path | str) -> list[ExtractedSection]:
    """Extract text from a plain text document."""
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    raw_text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw_text:
        return []
    return [ExtractedSection(page=1, location="Section 1", text=raw_text)]


def extract_document_sections(file_path: Path | str) -> list[ExtractedSection]:
    """Dispatch document text extraction based on file extension.

    Supports .pdf, .pptx, .docx, and .txt.
    """
    path = Path(file_path).resolve()
    ext = path.suffix.lower()

    if ext == ".pdf":
        return extract_pdf_sections(path)
    elif ext == ".pptx":
        return extract_pptx_sections(path)
    elif ext == ".docx":
        return extract_docx_sections(path)
    elif ext == ".txt":
        return extract_txt_sections(path)
    else:
        raise ValueError(f"Unsupported format '{ext}' for text extraction")


# ---------------------------------------------------------------------------
# 2. Configurable, Source-Aware Chunking
# ---------------------------------------------------------------------------

def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks preferring paragraph and sentence boundaries.

    Args:
        text: Input string to chunk.
        chunk_size: Maximum chunk length in characters (> 0).
        chunk_overlap: Overlap between consecutive chunks (>= 0 and < chunk_size).

    Returns:
        List of non-empty chunk strings.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")

    clean_text = text.strip()
    if not clean_text:
        return []
    if len(clean_text) <= chunk_size:
        return [clean_text]

    # Decompose into atomic units: paragraphs -> sentences -> words
    raw_paras = clean_text.split("\n\n")
    units: list[str] = []
    for p in raw_paras:
        p = p.strip()
        if not p:
            continue
        if len(p) <= chunk_size:
            units.append(p)
        else:
            # Paragraph exceeds chunk size, split by sentences
            sentences = re.split(r"(?<=[.!?])\s+", p)
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if len(s) <= chunk_size:
                    units.append(s)
                else:
                    # Sentence exceeds chunk size, split into word chunks
                    words = s.split()
                    cur_w: list[str] = []
                    cur_w_len = 0
                    for w in words:
                        w_len = len(w) + (1 if cur_w else 0)
                        if cur_w and cur_w_len + w_len > chunk_size:
                            units.append(" ".join(cur_w))
                            cur_w = [w]
                            cur_w_len = len(w)
                        else:
                            cur_w.append(w)
                            cur_w_len += w_len
                    if cur_w:
                        units.append(" ".join(cur_w))

    if not units:
        return []

    chunks: list[str] = []
    current_units: list[str] = []
    current_length = 0

    for u in units:
        add_len = len(u) + (2 if current_units else 0)
        if current_units and (current_length + add_len > chunk_size):
            chunk_str = "\n\n".join(current_units).strip()
            if chunk_str:
                chunks.append(chunk_str)

            # Build overlap from the end of current_units
            overlap_units: list[str] = []
            overlap_len = 0
            for prev_u in reversed(current_units):
                prev_add = len(prev_u) + (2 if overlap_units else 0)
                if overlap_len + prev_add <= chunk_overlap:
                    overlap_units.insert(0, prev_u)
                    overlap_len += prev_add
                else:
                    break

            current_units = overlap_units + [u]
            current_length = sum(len(x) for x in current_units) + 2 * max(0, len(current_units) - 1)
        else:
            current_units.append(u)
            current_length += add_len

    if current_units:
        final_str = "\n\n".join(current_units).strip()
        if final_str and (not chunks or final_str != chunks[-1]):
            chunks.append(final_str)

    return [c for c in chunks if c.strip()]


def chunk_sections(
    sections: list[ExtractedSection],
    file_id: int,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[DocumentChunk]:
    """Chunk extracted document sections into deterministic, source-mapped chunks.

    Args:
        sections: Extracted document sections.
        file_id: Registered file ID in database.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap in characters.

    Returns:
        List of DocumentChunk instances with deterministic IDs and chunk indexes.
    """
    chunks: list[DocumentChunk] = []
    global_index = 0

    for sec in sections:
        sec_text = sec.text.strip()
        if not sec_text:
            continue

        text_pieces = split_text_into_chunks(sec_text, chunk_size, chunk_overlap)
        for piece in text_pieces:
            chunk_id = f"chk_{file_id}_{global_index:04d}"
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    file_id=file_id,
                    page=sec.page,
                    location=sec.location,
                    chunk_index=global_index,
                    text=piece,
                )
            )
            global_index += 1

    return chunks


# ---------------------------------------------------------------------------
# 3. Persistence & Incremental Processing
# ---------------------------------------------------------------------------

def persist_chunks_for_file(
    file_id: int,
    chunks: list[DocumentChunk],
    db_path: Path | str | None = None,
) -> None:
    """Transactionally persist chunks for a file, updating FTS5 index via triggers.

    Replaces any existing chunks for this file cleanly.
    """
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        # 1. Delete previous chunks (triggers clean FTS virtual table automatically)
        cursor.execute("DELETE FROM chunks WHERE file_id = ?;", (file_id,))

        # 2. Insert new chunks (triggers insert into FTS virtual table automatically)
        cursor.executemany(
            """
            INSERT INTO chunks (id, file_id, page, text, location, chunk_index)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            [(c.id, c.file_id, c.page, c.text, c.location, c.chunk_index) for c in chunks],
        )

        # 3. Update file processing status
        cursor.execute(
            """
            UPDATE files
            SET status = 'processed',
                chunk_count = ?,
                error_message = NULL,
                processed_at = CURRENT_TIMESTAMP
            WHERE id = ?;
            """,
            (len(chunks), file_id),
        )


def process_file(
    file_id: int,
    force: bool = False,
    db_path: Path | str | None = None,
    settings: Settings | None = None,
) -> FileProcessingResult:
    """Incrementally extract and chunk a registered document.

    Args:
        file_id: Registered file ID in SQLite.
        force: If True, reprocesses even if already processed.
        db_path: Optional custom database path.
        settings: Optional custom Settings.

    Returns:
        FileProcessingResult detailing outcome and chunk count.
    """
    cfg = settings or load()
    f_row = get_file_by_id(file_id, db_path=db_path)
    if not f_row:
        raise ValueError(f"File ID {file_id} not found in database")

    filename = f_row["filename"]
    ext = (f_row["ext"] or "").lower()
    current_status = f_row["status"]
    file_path = Path(f_row["path"])

    # 1. Check idempotency: skip if already processed and not forced
    if current_status == "processed" and not force:
        existing_chunks = get_file_chunks(file_id, db_path=db_path)
        return FileProcessingResult(
            file_id=file_id,
            filename=filename,
            status="processed",
            chunk_count=len(existing_chunks),
        )

    # 2. Check format support
    if ext not in EXTRACTION_SUPPORTED_EXTENSIONS:
        msg = f"File format '{ext}' is not supported for text extraction"
        update_file_processing_status(
            file_id=file_id,
            status="unsupported",
            chunk_count=0,
            error_message=msg,
            db_path=db_path,
        )
        return FileProcessingResult(
            file_id=file_id,
            filename=filename,
            status="unsupported",
            chunk_count=0,
            error_message=msg,
        )

    # 3. Check physical file existence
    if not file_path.exists():
        msg = f"Source document file not found at path: {file_path}"
        update_file_processing_status(
            file_id=file_id,
            status="failed",
            chunk_count=0,
            error_message=msg,
            db_path=db_path,
        )
        return FileProcessingResult(
            file_id=file_id,
            filename=filename,
            status="failed",
            chunk_count=0,
            error_message=msg,
        )

    # 4. Extract document sections
    try:
        sections = extract_document_sections(file_path)
    except Exception as ex:
        err_msg = f"Text extraction failed: {ex}"
        logger.warning("Extraction failed for file %d (%s): %s", file_id, filename, ex)
        update_file_processing_status(
            file_id=file_id,
            status="failed",
            chunk_count=0,
            error_message=err_msg,
            db_path=db_path,
        )
        return FileProcessingResult(
            file_id=file_id,
            filename=filename,
            status="failed",
            chunk_count=0,
            error_message=err_msg,
        )

    # 5. Check if document produced any text (e.g. scanned image-only PDF)
    total_text_len = sum(len(s.text.strip()) for s in sections)
    if not sections or total_text_len == 0:
        msg = "No extractable text found (document may be scanned or image-only; requires OCR)"
        with db_session(db_path) as conn:
            # Cleanly remove old chunks if reprocessing an emptied document
            conn.execute("DELETE FROM chunks WHERE file_id = ?;", (file_id,))
            conn.execute(
                """
                UPDATE files
                SET status = 'empty',
                    chunk_count = 0,
                    error_message = ?,
                    processed_at = CURRENT_TIMESTAMP
                WHERE id = ?;
                """,
                (msg, file_id),
            )
        return FileProcessingResult(
            file_id=file_id,
            filename=filename,
            status="empty",
            chunk_count=0,
            error_message=msg,
        )

    # 6. Generate chunks
    chunks = chunk_sections(
        sections=sections,
        file_id=file_id,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    if not chunks:
        msg = "No chunks produced from extracted text"
        update_file_processing_status(
            file_id=file_id,
            status="empty",
            chunk_count=0,
            error_message=msg,
            db_path=db_path,
        )
        return FileProcessingResult(
            file_id=file_id,
            filename=filename,
            status="empty",
            chunk_count=0,
            error_message=msg,
        )

    # 7. Persist chunks atomically
    try:
        persist_chunks_for_file(file_id=file_id, chunks=chunks, db_path=db_path)
        return FileProcessingResult(
            file_id=file_id,
            filename=filename,
            status="processed",
            chunk_count=len(chunks),
        )
    except Exception as ex:
        err_msg = f"Failed to persist chunks: {ex}"
        logger.error("Persistence failed for file %d (%s): %s", file_id, filename, ex)
        update_file_processing_status(
            file_id=file_id,
            status="failed",
            error_message=err_msg,
            db_path=db_path,
        )
        return FileProcessingResult(
            file_id=file_id,
            filename=filename,
            status="failed",
            chunk_count=0,
            error_message=err_msg,
        )


def process_group_files(
    group_id: int | None = None,
    force: bool = False,
    db_path: Path | str | None = None,
    settings: Settings | None = None,
) -> ExtractionBatchResult:
    """Batch-process registered documents for a study group or across all groups.

    Args:
        group_id: Optional group ID. If None, processes across all groups.
        force: If True, reprocesses all files; if False, processes only pending/failed files.
        db_path: Custom database path.
        settings: Custom Settings.

    Returns:
        ExtractionBatchResult with counts and per-file results.
    """
    batch_res = ExtractionBatchResult()

    with db_session(db_path) as conn:
        cursor = conn.cursor()
        if group_id is not None:
            if force:
                cursor.execute(
                    "SELECT id, filename, status FROM files WHERE group_id = ? ORDER BY id ASC;",
                    (group_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, filename, status FROM files
                    WHERE group_id = ? AND status IN ('new', 'failed')
                    ORDER BY id ASC;
                    """,
                    (group_id,),
                )
        else:
            if force:
                cursor.execute("SELECT id, filename, status FROM files ORDER BY id ASC;")
            else:
                cursor.execute(
                    """
                    SELECT id, filename, status FROM files
                    WHERE status IN ('new', 'failed')
                    ORDER BY id ASC;
                    """
                )
        files_to_process = cursor.fetchall()

    batch_res.total_files = len(files_to_process)

    for f in files_to_process:
        f_id = f["id"]
        try:
            res = process_file(file_id=f_id, force=force, db_path=db_path, settings=settings)
            batch_res.results.append(res)
            if res.status == "processed":
                batch_res.processed_count += 1
            elif res.status == "empty":
                batch_res.empty_count += 1
            elif res.status == "unsupported":
                batch_res.unsupported_count += 1
            elif res.status == "failed":
                batch_res.failed_count += 1
        except Exception as ex:
            logger.error("Unexpected error processing file ID %d: %s", f_id, ex)
            fail_res = FileProcessingResult(
                file_id=f_id,
                filename=f["filename"],
                status="failed",
                chunk_count=0,
                error_message=str(ex),
            )
            batch_res.results.append(fail_res)
            batch_res.failed_count += 1

    return batch_res
