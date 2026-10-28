"""WhatsApp export and document folder ingestion pipeline for StudyRAG.

Handles ZIP archive extraction with ZIP Slip protection, recursive folder scanning,
SHA-256 deduplication, attachment-to-message linking, surrounding message context extraction,
persistent missing attachment tracking, and recovery workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
import re
import shutil
import sqlite3
from typing import Any
import unicodedata
import zipfile

from core.chat_parser import ChatMessage, find_chat_file, parse_chat
from core.config import load
from core.db import (
    db_session,
    get_attachment_ref,
    get_attachment_refs,
    get_file_by_id,
    get_file_by_sha256,
    get_group_files,
    get_or_create_group,
    insert_file,
    register_sender,
    update_attachment_ref_resolution,
    update_file_metadata,
    upsert_attachment_ref,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "IngestResult",
    "RecoveryResult",
    "compute_ref_dedup_key",
    "compute_sha256",
    "get_candidates_for_ref",
    "ingest_folder",
    "ingest_zip",
    "is_safe_zip_entry",
    "link_attachment_ref_manually",
    "link_files_to_chat",
    "normalize_filename",
    "normalize_input_path",
    "reconcile_attachment_refs",
    "recover_file_or_folder",
    "sanitize_filename",
    "validate_recovery_target",
]

# Supported document and image file extensions
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".pptx",
    ".docx",
    ".jpg",
    ".jpeg",
    ".png",
}

# Maximum allowed time difference (in seconds) for surrounding context messages (5 minutes)
SURROUNDING_WINDOW_SECONDS = 300
# Maximum number of surrounding context messages from the same sender
MAX_SURROUNDING_MESSAGES = 3


@dataclass
class IngestResult:
    """Summary of an ingestion run."""

    group_id: int
    group_name: str
    total_found: int = 0
    imported_count: int = 0
    duplicates_skipped: int = 0
    unsupported_count: int = 0
    failed_count: int = 0
    unmatched_chat_count: int = 0
    missing_attachments_count: int = 0
    ambiguous_attachments_count: int = 0
    registered_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    per_file_failures: dict[str, str] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    """Summary of an attachment recovery operation."""

    group_id: int
    group_name: str
    imported_files: list[str] = field(default_factory=list)
    duplicates_skipped: int = 0
    resolved_references: list[int] = field(default_factory=list)
    remaining_missing: int = 0
    ambiguous_references: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def sanitize_filename(name: str) -> str:
    """Sanitize directory or file names for filesystem compatibility."""
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip()


def normalize_filename(name: str | None) -> str:
    """Normalize a filename for safe, case-insensitive comparison.

    Applies Unicode NFKC normalization, strips whitespace, converts path separators,
    and returns lowercased basename.
    """
    if not name:
        return ""
    clean = unicodedata.normalize("NFKC", name.strip())
    clean = clean.replace("\\", "/")
    return Path(clean).name.lower()


def normalize_input_path(raw_path: str | Path, base_dir: Path | str | None = None) -> Path:
    """Safely normalize and resolve a user-entered file or folder path.

    Handles:
    - Stripping leading and trailing whitespace.
    - Stripping matching accidental surrounding single or double quotes (e.g. from Windows 'Copy as path').
    - Direct usage of absolute Windows paths without prepending project directory.
    - Resolution of relative paths against the explicitly intended base directory.
    - Preservation of spaces, parentheses, accents, and legitimate characters in filenames.
    """
    if isinstance(raw_path, Path):
        s = str(raw_path)
    else:
        s = str(raw_path) if raw_path is not None else ""

    s = s.strip()
    if not s:
        raise ValueError("Path cannot be empty.")

    # Remove matching surrounding single quotes
    if len(s) >= 2 and s.startswith("'") and s.endswith("'"):
        s = s[1:-1].strip()

    # Remove matching surrounding double quotes
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()

    # Double quotes are never valid in Windows paths or filenames.
    # Safely strip any stray leading/trailing double quotes from copy-paste glitches:
    s = s.strip('"').strip()

    # Re-check single quotes if they were wrapped inside double quotes
    if len(s) >= 2 and s.startswith("'") and s.endswith("'"):
        s = s[1:-1].strip()

    if not s:
        raise ValueError("Path cannot be empty.")

    p = Path(s)
    # Check absolute: standard is_absolute() or Windows drive letter pattern (e.g. C:\ or C:/)
    is_win_drive = bool(re.match(r"^[a-zA-Z]:[\\/]", s))
    if p.is_absolute() or is_win_drive:
        return p.resolve()

    base = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
    return (base / p).resolve()


def validate_recovery_target(
    target_path: Path | str,
    base_dir: Path | str | None = None,
) -> tuple[Path, str | None]:
    """Validate whether target_path exists, is accessible, and is a supported file or folder.

    Returns:
        (resolved_path, error_message).
        error_message is None if the path exists and is a supported file or directory.
    """
    try:
        path = normalize_input_path(target_path, base_dir=base_dir)
    except Exception as err:
        return Path(str(target_path)), f"Invalid path: {err}"

    if not path.exists():
        return path, f"Path does not exist: '{path}'"

    if path.is_file():
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            return path, f"File '{path.name}' has unsupported extension '{ext}'. Supported file types: {allowed}"
        return path, None

    if path.is_dir():
        return path, None

    return path, f"Path is not a regular file or directory: '{path}'"


def compute_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file using streaming chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_ref_dedup_key(
    group_id: int,
    sender: str | None,
    sent_at: str | None,
    filename: str | None,
    text: str,
) -> str:
    """Compute a deterministic, stable deduplication key for an attachment reference."""
    norm_fn = normalize_filename(filename) if filename else ""
    seed = f"{group_id}|{sender or ''}|{sent_at or ''}|{norm_fn}|{text[:150].strip()}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def is_safe_zip_entry(target_dir: Path, entry_name: str) -> bool:
    """Verify whether a ZIP entry path is safe from ZIP Slip and path traversal.

    Rejects absolute paths, Windows drive references, and traversal outside target_dir.
    """
    clean_name = entry_name.replace("\\", "/")
    if clean_name.startswith("/") or clean_name.startswith("\\"):
        return False
    if len(clean_name) > 1 and clean_name[1] == ":":
        return False

    try:
        resolved_target = target_dir.resolve()
        dest = (target_dir / entry_name).resolve()
        return dest.is_relative_to(resolved_target)
    except (ValueError, RuntimeError):
        return False


def build_file_caption(
    attachment_msg: ChatMessage,
    all_messages: list[ChatMessage],
) -> str | None:
    """Build caption combining the attachment message and surrounding messages from the same sender.

    Rules:
    - Same sender only.
    - Within 5 minutes before or after the attachment message.
    - At most 3 surrounding messages in addition to the attachment message.
    - Preserves attachment message's own caption even if no surrounding messages match.
    """
    att_time = attachment_msg.datetime_obj
    att_sender = attachment_msg.sender

    base_text = attachment_msg.text.strip()
    if not att_time or not att_sender:
        return base_text if base_text else None

    candidates: list[tuple[float, ChatMessage]] = []
    for m in all_messages:
        if m is attachment_msg:
            continue
        if m.sender != att_sender or not m.datetime_obj:
            continue
        diff_sec = abs((m.datetime_obj - att_time).total_seconds())
        if diff_sec <= SURROUNDING_WINDOW_SECONDS and m.text.strip():
            candidates.append((diff_sec, m))

    candidates.sort(key=lambda item: item[0])
    selected_surrounding = [item[1] for item in candidates[:MAX_SURROUNDING_MESSAGES]]

    all_selected = selected_surrounding + [attachment_msg]
    all_selected.sort(key=lambda m: m.datetime_obj if m.datetime_obj else att_time)

    caption_parts: list[str] = []
    for m in all_selected:
        t = m.text.strip()
        if t and t not in caption_parts:
            caption_parts.append(t)

    return "\n".join(caption_parts) if caption_parts else None


def link_files_to_chat(
    discovered_files: list[Path],
    chat_messages: list[ChatMessage],
    result: IngestResult,
    group_id: int,
    db_path: Path | str | None = None,
    unsupported_entries: list[str] | None = None,
    per_file_failures: dict[str, str] | None = None,
) -> dict[Path, tuple[str | None, str | None, str | None]]:
    """Match discovered documents to chat attachment messages and persist attachment references.

    Tracks missing, ambiguous, available, unsupported, and failed attachments in SQLite.

    Returns:
        Mapping of document Path -> (sender, sent_at, caption)
    """
    unsupported_set = {normalize_filename(p) for p in (unsupported_entries or [])}
    failed_set = {normalize_filename(p) for p in (per_file_failures or {}).keys()}

    # Group attachment messages by normalized filename
    att_messages: list[ChatMessage] = []
    att_map: dict[str, list[ChatMessage]] = {}
    for m in chat_messages:
        if m.file or m.has_attachment:
            att_messages.append(m)
            if m.file:
                norm_fn = normalize_filename(m.file)
                att_map.setdefault(norm_fn, []).append(m)

    # Discovered files count by normalized name in this import
    doc_name_counts: dict[str, int] = {}
    for doc in discovered_files:
        norm = normalize_filename(doc.name)
        doc_name_counts[norm] = doc_name_counts.get(norm, 0) + 1

    # Check previously registered files in the group
    existing_group_files = get_group_files(group_id, db_path=db_path)
    existing_file_map: dict[str, list[sqlite3.Row]] = {}
    for ef in existing_group_files:
        norm = normalize_filename(ef["filename"])
        existing_file_map.setdefault(norm, []).append(ef)

    file_metadata: dict[Path, tuple[str | None, str | None, str | None]] = {}

    # 1. Match discovered files to single unambiguous messages
    for doc in discovered_files:
        norm_doc_name = normalize_filename(doc.name)
        matching_msgs = att_map.get(norm_doc_name, [])

        if not matching_msgs:
            file_metadata[doc] = (None, None, None)
            result.unmatched_chat_count += 1
            continue

        if len(matching_msgs) > 1 or doc_name_counts.get(norm_doc_name, 0) > 1:
            result.warnings.append(
                f"Ambiguous attachment '{doc.name}': multiple matching messages or duplicate files found. "
                "Skipping automated message-caption assignment to prevent mismatched metadata."
            )
            file_metadata[doc] = (None, None, None)
            continue

        att_msg = matching_msgs[0]
        caption = build_file_caption(att_msg, chat_messages)
        file_metadata[doc] = (att_msg.sender, att_msg.sent_at, caption)

    # 2. Persist attachment references in SQLite
    for m in att_messages:
        orig_fn = m.file
        norm_fn = normalize_filename(orig_fn) if orig_fn else ""
        context = build_file_caption(m, chat_messages)
        dedup_key = compute_ref_dedup_key(group_id, m.sender, m.sent_at, orig_fn, m.text)

        # Case A: Media omitted without filename
        if not orig_fn:
            upsert_attachment_ref(
                group_id=group_id,
                dedup_key=dedup_key,
                status="missing",
                original_filename=None,
                normalized_filename=None,
                sender=m.sender,
                sent_at=m.sent_at,
                caption=m.text,
                context=context,
                resolution_method="unresolved",
                resolution_note="Media omitted by WhatsApp in export without filename; message context preserved.",
                db_path=db_path,
            )
            result.missing_attachments_count += 1
            continue

        # Case B: Referenced file is unsupported format
        if norm_fn in unsupported_set:
            upsert_attachment_ref(
                group_id=group_id,
                dedup_key=dedup_key,
                status="unsupported",
                original_filename=orig_fn,
                normalized_filename=norm_fn,
                sender=m.sender,
                sent_at=m.sent_at,
                caption=m.text,
                context=context,
                resolution_method="unresolved",
                resolution_note=f"Attachment '{orig_fn}' exists in export but has an unsupported file format.",
                db_path=db_path,
            )
            continue

        # Case C: Referenced file failed extraction
        if norm_fn in failed_set:
            fail_reason = per_file_failures.get(orig_fn, "Unknown failure") if per_file_failures else "Failed"
            upsert_attachment_ref(
                group_id=group_id,
                dedup_key=dedup_key,
                status="import_failed",
                original_filename=orig_fn,
                normalized_filename=norm_fn,
                sender=m.sender,
                sent_at=m.sent_at,
                caption=m.text,
                context=context,
                resolution_method="unresolved",
                resolution_note=f"Attachment '{orig_fn}' failed during extraction: {fail_reason}",
                db_path=db_path,
            )
            continue

        # Case D: Check candidates among discovered documents and existing group files
        count_discovered = doc_name_counts.get(norm_fn, 0)
        count_existing = len(existing_file_map.get(norm_fn, []))
        matching_chat_msgs = att_map.get(norm_fn, [])

        if len(matching_chat_msgs) > 1 or (count_discovered + count_existing) > 1:
            upsert_attachment_ref(
                group_id=group_id,
                dedup_key=dedup_key,
                status="ambiguous",
                original_filename=orig_fn,
                normalized_filename=norm_fn,
                sender=m.sender,
                sent_at=m.sent_at,
                caption=m.text,
                context=context,
                resolution_method="unresolved",
                resolution_note=f"Multiple candidate files or messages match '{orig_fn}'; manual resolution required.",
                db_path=db_path,
            )
            result.ambiguous_attachments_count += 1
            continue

        if count_discovered == 1 or count_existing == 1:
            # File is available!
            matched_file_id: int | None = None
            if count_existing == 1:
                matched_file_id = existing_file_map[norm_fn][0]["id"]

            upsert_attachment_ref(
                group_id=group_id,
                dedup_key=dedup_key,
                status="available",
                original_filename=orig_fn,
                normalized_filename=norm_fn,
                sender=m.sender,
                sent_at=m.sent_at,
                caption=m.text,
                context=context,
                file_id=matched_file_id,
                resolution_method="automatic",
                resolution_note="Auto-matched by exact filename",
                db_path=db_path,
            )
            continue

        # Case E: File not found on disk or in DB -> Missing!
        upsert_attachment_ref(
            group_id=group_id,
            dedup_key=dedup_key,
            status="missing",
            original_filename=orig_fn,
            normalized_filename=norm_fn,
            sender=m.sender,
            sent_at=m.sent_at,
            caption=m.text,
            context=context,
            file_id=None,
            resolution_method="unresolved",
            resolution_note=f"Attachment '{orig_fn}' referenced in chat by '{m.sender}' was not found in export or group storage.",
            db_path=db_path,
        )
        result.missing_attachments_count += 1
        result.warnings.append(
            f"Attachment '{orig_fn}' mentioned in chat by '{m.sender}' was not found in export."
        )

    return file_metadata


def ingest_zip(
    zip_path: Path | str,
    group_name: str,
    db_path: Path | str | None = None,
    data_dir: Path | str | None = None,
) -> IngestResult:
    """Ingest a WhatsApp exported ZIP archive.

    Extracts supported documents with ZIP Slip protection, parses chat logs,
    tracks missing and available attachment references, computes SHA-256 hashes,
    deduplicates within the group, and records metadata in SQLite.
    """
    clean_group = group_name.strip()
    if not clean_group:
        raise ValueError("Group name cannot be empty")

    source_zip = normalize_input_path(zip_path)
    if not source_zip.exists():
        raise FileNotFoundError(f"ZIP file not found at '{source_zip}'")

    base_data_dir = Path(data_dir).resolve() if data_dir else load().resolved_data_dir
    group_dir = base_data_dir / sanitize_filename(clean_group)
    group_dir.mkdir(parents=True, exist_ok=True)

    group_id = get_or_create_group(clean_group, db_path=db_path)
    result = IngestResult(group_id=group_id, group_name=clean_group)

    if not zipfile.is_zipfile(source_zip):
        err = f"File '{source_zip}' is not a valid ZIP archive"
        result.errors.append(err)
        return result

    extracted_files: list[Path] = []
    unsupported_entries: list[str] = []

    try:
        with zipfile.ZipFile(source_zip, "r") as zf:
            for zip_info in zf.infolist():
                if zip_info.is_dir():
                    continue

                entry_name = zip_info.filename

                if not is_safe_zip_entry(group_dir, entry_name):
                    result.warnings.append(
                        f"Unsafe ZIP entry path rejected (ZIP Slip defense): '{entry_name}'"
                    )
                    result.failed_count += 1
                    result.per_file_failures[entry_name] = "Unsafe path traversal rejected"
                    continue

                entry_path = Path(entry_name)
                ext = entry_path.suffix.lower()

                is_chat = ext == ".txt"
                is_supported = ext in SUPPORTED_EXTENSIONS

                if not is_chat and not is_supported:
                    unsupported_entries.append(entry_name)
                    continue

                dest_file = group_dir / entry_path.name

                if dest_file.exists():
                    try:
                        existing_hash = compute_sha256(dest_file)
                        archive_data = zf.read(zip_info)
                        archive_hash = hashlib.sha256(archive_data).hexdigest()

                        if existing_hash != archive_hash:
                            counter = 1
                            while dest_file.exists():
                                dest_file = group_dir / f"{entry_path.stem}_{counter}{entry_path.suffix}"
                                counter += 1
                            dest_file.write_bytes(archive_data)
                    except Exception as ex:
                        result.per_file_failures[entry_name] = f"Extraction collision error: {ex}"
                        result.failed_count += 1
                        continue
                else:
                    try:
                        with zf.open(zip_info) as source, open(dest_file, "wb") as target:
                            shutil.copyfileobj(source, target)
                    except Exception as ex:
                        result.per_file_failures[entry_name] = f"Extraction error: {ex}"
                        result.failed_count += 1
                        continue

                extracted_files.append(dest_file)

    except (zipfile.BadZipFile, OSError) as err:
        result.errors.append(f"Failed to extract ZIP archive: {err}")
        return result

    result.unsupported_count = len(unsupported_entries)

    # 1. Parse WhatsApp chat if present
    chat_file = find_chat_file(group_dir)
    chat_messages: list[ChatMessage] = []
    if chat_file and chat_file.exists():
        try:
            chat_text = chat_file.read_text(encoding="utf-8", errors="replace")
            chat_messages = parse_chat(chat_text)

            for m in chat_messages:
                if m.sender:
                    register_sender(group_id, m.sender, db_path=db_path)
        except Exception as ex:
            result.warnings.append(f"Failed to parse chat log '{chat_file.name}': {ex}")

    # 2. Identify supported study documents
    discovered_docs = [
        p for p in extracted_files if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    result.total_found = len(discovered_docs)

    # 3. Link documents and record persistent attachment references
    metadata_map = link_files_to_chat(
        discovered_files=discovered_docs,
        chat_messages=chat_messages,
        result=result,
        group_id=group_id,
        db_path=db_path,
        unsupported_entries=unsupported_entries,
        per_file_failures=result.per_file_failures,
    )

    if result.total_found == 0:
        result.warnings.append(
            "Export contains zero supported documents (.pdf, .pptx, .docx, images). "
            "WhatsApp exports without media omit documents. Please export the chat with media "
            "or use folder import."
        )
        return result

    # 4. Deduplicate and register files in SQLite
    for doc in discovered_docs:
        try:
            doc_sha256 = compute_sha256(doc)
            existing_row = get_file_by_sha256(group_id, doc_sha256, db_path=db_path)

            if existing_row:
                result.duplicates_skipped += 1
                file_id = existing_row["id"]
            else:
                sender, sent_at, caption = metadata_map.get(doc, (None, None, None))
                file_id = insert_file(
                    group_id=group_id,
                    sha256=doc_sha256,
                    filename=doc.name,
                    path=str(doc.resolve()),
                    ext=doc.suffix.lower(),
                    sender=sender,
                    sent_at=sent_at,
                    caption=caption,
                    status="new",
                    db_path=db_path,
                )
                result.imported_count += 1
                result.registered_files.append(str(doc.resolve()))

            # Link available attachment references to this newly registered file_id
            norm_doc = normalize_filename(doc.name)
            with db_session(db_path) as conn:
                conn.execute(
                    """
                    UPDATE attachment_refs
                    SET file_id = ?
                    WHERE group_id = ? AND normalized_filename = ? AND status = 'available' AND file_id IS NULL;
                    """,
                    (file_id, group_id, norm_doc),
                )

        except Exception as ex:
            result.failed_count += 1
            result.per_file_failures[doc.name] = str(ex)

    return result


def ingest_folder(
    folder_path: Path | str,
    group_name: str,
    db_path: Path | str | None = None,
    data_dir: Path | str | None = None,
) -> IngestResult:
    """Ingest documents from a local directory.

    Recursively discovers supported files, copies them non-destructively to the group
    directory, links chat logs if present, deduplicates using SHA-256, and records in SQLite.
    """
    clean_group = group_name.strip()
    if not clean_group:
        raise ValueError("Group name cannot be empty")

    source_dir = normalize_input_path(folder_path)
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Folder not found at '{source_dir}'")

    base_data_dir = Path(data_dir).resolve() if data_dir else load().resolved_data_dir
    group_dir = base_data_dir / sanitize_filename(clean_group)
    group_dir.mkdir(parents=True, exist_ok=True)

    group_id = get_or_create_group(clean_group, db_path=db_path)
    result = IngestResult(group_id=group_id, group_name=clean_group)

    discovered_supported: list[Path] = []
    unsupported_count = 0

    try:
        all_entries = list(source_dir.rglob("*"))
    except OSError as err:
        result.errors.append(f"Cannot scan folder '{source_dir}': {err}")
        return result

    for p in all_entries:
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            discovered_supported.append(p)
        elif ext != ".txt":
            unsupported_count += 1

    result.unsupported_count = unsupported_count
    result.total_found = len(discovered_supported)

    chat_file = find_chat_file(source_dir) or find_chat_file(group_dir)
    chat_messages: list[ChatMessage] = []
    if chat_file and chat_file.exists():
        try:
            chat_text = chat_file.read_text(encoding="utf-8", errors="replace")
            chat_messages = parse_chat(chat_text)
            for m in chat_messages:
                if m.sender:
                    register_sender(group_id, m.sender, db_path=db_path)
        except Exception as ex:
            result.warnings.append(f"Failed to parse chat log in folder: {ex}")

    copied_docs: list[Path] = []
    for src_file in discovered_supported:
        try:
            if src_file.parent.resolve() == group_dir.resolve():
                dest_file = src_file
            else:
                dest_file = group_dir / src_file.name
                if dest_file.exists():
                    src_hash = compute_sha256(src_file)
                    dest_hash = compute_sha256(dest_file)
                    if src_hash != dest_hash:
                        counter = 1
                        while dest_file.exists():
                            dest_file = group_dir / f"{src_file.stem}_{counter}{src_file.suffix}"
                            counter += 1
                        shutil.copy2(src_file, dest_file)
                else:
                    shutil.copy2(src_file, dest_file)

            copied_docs.append(dest_file)

        except Exception as ex:
            result.failed_count += 1
            result.per_file_failures[src_file.name] = f"Copy error: {ex}"

    metadata_map = link_files_to_chat(
        discovered_files=copied_docs,
        chat_messages=chat_messages,
        result=result,
        group_id=group_id,
        db_path=db_path,
        unsupported_entries=[],
        per_file_failures=result.per_file_failures,
    )

    if result.total_found == 0:
        result.warnings.append(
            "Folder contains zero supported documents (.pdf, .pptx, .docx, images)."
        )
        return result

    for doc in copied_docs:
        try:
            doc_sha256 = compute_sha256(doc)
            existing_row = get_file_by_sha256(group_id, doc_sha256, db_path=db_path)

            if existing_row:
                result.duplicates_skipped += 1
                file_id = existing_row["id"]
            else:
                sender, sent_at, caption = metadata_map.get(doc, (None, None, None))
                file_id = insert_file(
                    group_id=group_id,
                    sha256=doc_sha256,
                    filename=doc.name,
                    path=str(doc.resolve()),
                    ext=doc.suffix.lower(),
                    sender=sender,
                    sent_at=sent_at,
                    caption=caption,
                    status="new",
                    db_path=db_path,
                )
                result.imported_count += 1
                result.registered_files.append(str(doc.resolve()))

            norm_doc = normalize_filename(doc.name)
            with db_session(db_path) as conn:
                conn.execute(
                    """
                    UPDATE attachment_refs
                    SET file_id = ?
                    WHERE group_id = ? AND normalized_filename = ? AND status = 'available' AND file_id IS NULL;
                    """,
                    (file_id, group_id, norm_doc),
                )

        except Exception as ex:
            result.failed_count += 1
            result.per_file_failures[doc.name] = str(ex)

    return result


def reconcile_attachment_refs(
    group_id: int,
    db_path: Path | str | None = None,
) -> dict[str, int]:
    """Reconcile unresolved (missing or ambiguous) attachment references against registered files.

    Performs safe exact and normalized filename matching. Multiple candidates or duplicate
    references transition to 'ambiguous'.

    Returns:
        dict with counts of newly resolved, ambiguous, and remaining missing references.
    """
    counts = {"resolved": 0, "ambiguous": 0, "remaining_missing": 0}
    unresolved_refs = get_attachment_refs(group_id=group_id, db_path=db_path)
    candidates_to_check = [
        r for r in unresolved_refs if r["status"] in ("missing", "ambiguous")
    ]
    if not candidates_to_check:
        return counts

    group_files = get_group_files(group_id=group_id, db_path=db_path)
    if not group_files:
        counts["remaining_missing"] = len(candidates_to_check)
        return counts

    file_map: dict[str, list[sqlite3.Row]] = {}
    for f in group_files:
        norm = normalize_filename(f["filename"])
        file_map.setdefault(norm, []).append(f)

    ref_name_counts: dict[str, int] = {}
    for r in candidates_to_check:
        norm = r["normalized_filename"]
        if norm:
            ref_name_counts[norm] = ref_name_counts.get(norm, 0) + 1

    for ref in candidates_to_check:
        ref_id = ref["id"]
        norm = ref["normalized_filename"]
        if not norm:
            counts["remaining_missing"] += 1
            continue

        matching_files = file_map.get(norm, [])

        if len(matching_files) == 1 and ref_name_counts.get(norm, 0) == 1:
            matched_file = matching_files[0]
            update_attachment_ref_resolution(
                ref_id=ref_id,
                file_id=matched_file["id"],
                status="available",
                method="automatic",
                note="Resolved via recovery import (exact filename match)",
                db_path=db_path,
            )
            update_file_metadata(
                file_id=matched_file["id"],
                sender=ref["sender"],
                sent_at=ref["sent_at"],
                caption=ref["context"] or ref["caption"],
                db_path=db_path,
            )
            counts["resolved"] += 1
        elif len(matching_files) > 1 or ref_name_counts.get(norm, 0) > 1:
            update_attachment_ref_resolution(
                ref_id=ref_id,
                file_id=None,
                status="ambiguous",
                method="unresolved",
                note=f"Multiple candidate files ({len(matching_files)}) or references match '{ref['original_filename']}'; manual resolution required.",
                db_path=db_path,
            )
            counts["ambiguous"] += 1
        else:
            counts["remaining_missing"] += 1

    return counts


def recover_file_or_folder(
    target_path: Path | str,
    group_id: int,
    db_path: Path | str | None = None,
    data_dir: Path | str | None = None,
) -> RecoveryResult:
    """Import a downloaded file or folder to recover missing attachment references.

    Idempotently registers new files, preserves original files, and reconciles
    unresolved attachment references in the specified group.
    """
    path_obj = normalize_input_path(target_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Recovery target path does not exist: '{path_obj}'")

    with db_session(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM groups WHERE id = ?;", (group_id,))
        grow = cur.fetchone()
        if not grow:
            raise ValueError(f"Group ID {group_id} not found in database")
        group_name = grow["name"]

    res = RecoveryResult(group_id=group_id, group_name=group_name)

    base_data_dir = Path(data_dir).resolve() if data_dir else load().resolved_data_dir
    group_dir = base_data_dir / sanitize_filename(group_name)
    group_dir.mkdir(parents=True, exist_ok=True)

    if path_obj.is_file():
        ext = path_obj.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            res.errors.append(f"File '{path_obj.name}' has unsupported extension '{ext}'")
            return res

        if path_obj.parent.resolve() == group_dir.resolve():
            dest_file = path_obj
        else:
            dest_file = group_dir / path_obj.name
            if dest_file.exists():
                src_hash = compute_sha256(path_obj)
                dest_hash = compute_sha256(dest_file)
                if src_hash != dest_hash:
                    counter = 1
                    while dest_file.exists():
                        dest_file = group_dir / f"{path_obj.stem}_{counter}{path_obj.suffix}"
                        counter += 1
                    shutil.copy2(path_obj, dest_file)
            else:
                shutil.copy2(path_obj, dest_file)

        doc_sha256 = compute_sha256(dest_file)
        existing_row = get_file_by_sha256(group_id, doc_sha256, db_path=db_path)
        if existing_row:
            res.duplicates_skipped += 1
        else:
            insert_file(
                group_id=group_id,
                sha256=doc_sha256,
                filename=dest_file.name,
                path=str(dest_file.resolve()),
                ext=dest_file.suffix.lower(),
                status="new",
                db_path=db_path,
            )
            res.imported_files.append(str(dest_file.resolve()))

    elif path_obj.is_dir():
        folder_res = ingest_folder(
            folder_path=path_obj,
            group_name=group_name,
            db_path=db_path,
            data_dir=data_dir,
        )
        res.imported_files = folder_res.registered_files
        res.duplicates_skipped = folder_res.duplicates_skipped
        res.warnings.extend(folder_res.warnings)
        res.errors.extend(folder_res.errors)

    # Reconcile references
    reconcile_counts = reconcile_attachment_refs(group_id=group_id, db_path=db_path)
    res.remaining_missing = reconcile_counts["remaining_missing"]
    res.ambiguous_references = reconcile_counts["ambiguous"]

    available_refs = get_attachment_refs(group_id=group_id, status="available", db_path=db_path)
    res.resolved_references = [int(r["id"]) for r in available_refs]

    return res


def link_attachment_ref_manually(
    ref_id: int,
    file_id: int,
    db_path: Path | str | None = None,
) -> tuple[bool, str]:
    """Manually link an attachment reference to a registered file.

    Validates that both exist and belong to the same group, updates status to
    'available', records 'manual' resolution method, and links metadata.
    """
    ref = get_attachment_ref(ref_id, db_path=db_path)
    if not ref:
        return False, f"Attachment reference with ID {ref_id} not found."

    file_row = get_file_by_id(file_id, db_path=db_path)
    if not file_row:
        return False, f"File with ID {file_id} not found."

    if ref["group_id"] != file_row["group_id"]:
        return (
            False,
            f"Cross-group linking disallowed: reference group ({ref['group_id']}) "
            f"does not match file group ({file_row['group_id']}).",
        )

    note = f"Manually linked by user to '{file_row['filename']}' (File ID: {file_id})"
    update_attachment_ref_resolution(
        ref_id=ref_id,
        file_id=file_id,
        status="available",
        method="manual",
        note=note,
        db_path=db_path,
    )

    caption_to_set = ref["context"] or ref["caption"]
    update_file_metadata(
        file_id=file_id,
        sender=ref["sender"],
        sent_at=ref["sent_at"],
        caption=caption_to_set,
        db_path=db_path,
    )

    return True, f"Successfully linked reference to '{file_row['filename']}' (File ID: {file_id})."


def get_candidates_for_ref(
    ref_id: int,
    db_path: Path | str | None = None,
) -> list[sqlite3.Row]:
    """Retrieve candidate registered files in the group for an unresolved reference."""
    ref = get_attachment_ref(ref_id, db_path=db_path)
    if not ref:
        return []
    group_id = ref["group_id"]
    all_files = get_group_files(group_id, db_path=db_path)
    norm = ref["normalized_filename"]

    if not norm:
        return all_files

    exact_matches = [f for f in all_files if normalize_filename(f["filename"]) == norm]
    if exact_matches:
        return exact_matches

    return all_files
