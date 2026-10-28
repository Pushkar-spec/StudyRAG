"""WhatsApp conversational context retrieval and context-aware filtering for StudyRAG.

Retrieves sharing message metadata, timestamps, senders, and surrounding conversation
linked to imported study documents, and provides composable filters for search pipelines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
import sqlite3
from typing import Any

from core.db import (
    db_session,
    get_date_bounds,
    get_distinct_senders,
    get_files_whatsapp_context,
)

logger = logging.getLogger(__name__)

__all__ = [
    "WhatsAppMessageContext",
    "DocumentWhatsAppContext",
    "WhatsAppFilterCriteria",
    "FilterEvaluationResult",
    "get_whatsapp_context_for_documents",
    "get_whatsapp_context_for_file",
    "evaluate_document_filter",
    "filter_ranked_documents",
    "get_available_senders",
    "get_available_date_range",
]


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass
class WhatsAppMessageContext:
    """A single WhatsApp message or attachment reference linking to a study document."""

    ref_id: int | None
    sender: str | None
    sent_at: str | None
    message_text: str | None  # The direct message / caption accompanying the attachment
    surrounding_context: str | None  # Context window containing surrounding messages
    group_id: int
    group_name: str
    original_filename: str | None
    status: str = "available"
    resolution_method: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to standard dictionary."""
        return asdict(self)


@dataclass
class DocumentWhatsAppContext:
    """All WhatsApp conversational context associated with an imported study document."""

    file_id: int
    filename: str
    group_id: int
    group_name: str
    has_linked_message: bool
    primary_message: WhatsAppMessageContext | None = None
    all_messages: list[WhatsAppMessageContext] = field(default_factory=list)

    @property
    def sender(self) -> str | None:
        """Convenience property for primary sender."""
        return self.primary_message.sender if self.primary_message else None

    @property
    def sent_at(self) -> str | None:
        """Convenience property for primary sent_at timestamp."""
        return self.primary_message.sent_at if self.primary_message else None

    @property
    def message_text(self) -> str | None:
        """Convenience property for primary direct message text."""
        return self.primary_message.message_text if self.primary_message else None

    @property
    def surrounding_context(self) -> str | None:
        """Convenience property for primary surrounding conversation context."""
        return self.primary_message.surrounding_context if self.primary_message else None

    def to_dict(self) -> dict[str, Any]:
        """Convert to standard dictionary representation."""
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "group_id": self.group_id,
            "group_name": self.group_name,
            "has_linked_message": self.has_linked_message,
            "sender": self.sender,
            "sent_at": self.sent_at,
            "message_text": self.message_text,
            "surrounding_context": self.surrounding_context,
            "primary_message": self.primary_message.to_dict() if self.primary_message else None,
            "all_messages": [m.to_dict() for m in self.all_messages],
        }


@dataclass
class WhatsAppFilterCriteria:
    """Optional context-aware filtering constraints for search results."""

    group_ids: list[int] | None = None
    senders: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None
    has_linked_message: bool | None = None
    message_query: str | None = None

    def is_active(self) -> bool:
        """Return True if any filter constraint is enabled."""
        if self.group_ids is not None and len(self.group_ids) > 0:
            return True
        if self.senders is not None and len(self.senders) > 0:
            return True
        if self.date_from and self.date_from.strip():
            return True
        if self.date_to and self.date_to.strip():
            return True
        if self.has_linked_message is not None:
            return True
        if self.message_query and self.message_query.strip():
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "group_ids": self.group_ids,
            "senders": self.senders,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "has_linked_message": self.has_linked_message,
            "message_query": self.message_query,
            "is_active": self.is_active(),
        }


@dataclass
class FilterEvaluationResult:
    """Outcome of evaluating a document against WhatsApp filter criteria."""

    passed: bool
    reason: str
    is_missing_metadata: bool = False


# ---------------------------------------------------------------------------
# Context Retrieval
# ---------------------------------------------------------------------------

def get_whatsapp_context_for_documents(
    file_ids: list[int],
    db_path: Path | str | None = None,
) -> dict[int, DocumentWhatsAppContext]:
    """Retrieve WhatsApp conversational context for a list of document IDs.

    Uses the existing relationship between files, groups, and attachment_refs in SQLite.
    Never reparses ZIPs or invents missing data.

    Args:
        file_ids: List of integer file IDs.
        db_path: Optional custom SQLite database path.

    Returns:
        Mapping of file_id -> DocumentWhatsAppContext.
    """
    if not file_ids:
        return {}

    try:
        rows = get_files_whatsapp_context(file_ids, db_path=db_path)
    except Exception as err:
        logger.warning("Failed to retrieve WhatsApp context for file_ids %s: %s", file_ids, err)
        return {}

    context_map: dict[int, DocumentWhatsAppContext] = {}

    for r in rows:
        fid = int(r["file_id"])
        fn = str(r["file_filename"])
        gid = int(r["file_group_id"])
        gname = str(r["group_name"] or f"Group #{gid}")

        if fid not in context_map:
            # Check if files table itself has sender or caption (fallback if no attachment_refs)
            file_sender = str(r["file_sender"]) if r["file_sender"] else None
            file_sent_at = str(r["file_sent_at"]) if r["file_sent_at"] else None
            file_caption = str(r["file_caption"]) if r["file_caption"] else None

            context_map[fid] = DocumentWhatsAppContext(
                file_id=fid,
                filename=fn,
                group_id=gid,
                group_name=gname,
                has_linked_message=False,
                primary_message=None,
                all_messages=[],
            )

            # If files table has metadata but attachment_refs row is missing, populate provisional message
            if file_sender or file_sent_at or file_caption:
                context_map[fid].has_linked_message = True
                msg = WhatsAppMessageContext(
                    ref_id=None,
                    sender=file_sender,
                    sent_at=file_sent_at,
                    message_text=file_caption,
                    surrounding_context=file_caption,
                    group_id=gid,
                    group_name=gname,
                    original_filename=fn,
                    status="available",
                    resolution_method="direct_file_record",
                )
                context_map[fid].primary_message = msg
                context_map[fid].all_messages.append(msg)

        # Process joined attachment_refs row if present
        ref_id = r["ref_id"]
        if ref_id is not None:
            ref_sender = str(r["ref_sender"]) if r["ref_sender"] else None
            ref_sent_at = str(r["ref_sent_at"]) if r["ref_sent_at"] else None
            ref_caption = str(r["ref_caption"]) if r["ref_caption"] else None
            ref_context = str(r["ref_context"]) if r["ref_context"] else None
            ref_orig_fn = str(r["ref_original_filename"]) if r["ref_original_filename"] else fn
            ref_status = str(r["ref_status"] or "available")
            ref_res_method = str(r["ref_resolution_method"]) if r["ref_resolution_method"] else None

            msg = WhatsAppMessageContext(
                ref_id=int(ref_id),
                sender=ref_sender,
                sent_at=ref_sent_at,
                message_text=ref_caption,
                surrounding_context=ref_context,
                group_id=gid,
                group_name=gname,
                original_filename=ref_orig_fn,
                status=ref_status,
                resolution_method=ref_res_method,
            )

            doc_ctx = context_map[fid]
            doc_ctx.has_linked_message = True

            # If primary message was only a provisional direct_file_record, replace with authoritative ref
            if (
                doc_ctx.primary_message is None
                or doc_ctx.primary_message.ref_id is None
            ):
                doc_ctx.primary_message = msg
                # Remove provisional message if present
                doc_ctx.all_messages = [m for m in doc_ctx.all_messages if m.ref_id is not None]

            # Avoid adding duplicate ref_ids
            if not any(m.ref_id == int(ref_id) for m in doc_ctx.all_messages):
                doc_ctx.all_messages.append(msg)

    # For any requested file_ids that returned zero rows, create empty unlinked context
    for fid in file_ids:
        if fid not in context_map:
            context_map[fid] = DocumentWhatsAppContext(
                file_id=fid,
                filename=f"file_{fid}",
                group_id=0,
                group_name="Unknown Group",
                has_linked_message=False,
                primary_message=None,
                all_messages=[],
            )

    return context_map


def get_whatsapp_context_for_file(
    file_id: int,
    db_path: Path | str | None = None,
) -> DocumentWhatsAppContext | None:
    """Retrieve WhatsApp conversational context for a single document ID."""
    results = get_whatsapp_context_for_documents([file_id], db_path=db_path)
    return results.get(file_id)


# ---------------------------------------------------------------------------
# Context-Aware Filtering
# ---------------------------------------------------------------------------

def _extract_date_str(val: str | None) -> str | None:
    """Safely extract YYYY-MM-DD from an ISO timestamp or date string."""
    if not val:
        return None
    clean = val.strip()
    if len(clean) >= 10:
        return clean[:10]
    return clean


def evaluate_document_filter(
    context: DocumentWhatsAppContext,
    criteria: WhatsAppFilterCriteria,
) -> FilterEvaluationResult:
    """Evaluate whether a document's WhatsApp context satisfies the filter criteria.

    Handles missing metadata explicitly without silently treating absent metadata as a match.
    """
    if not criteria.is_active():
        return FilterEvaluationResult(passed=True, reason="No active filters")

    # 1. Group ID Filter
    if criteria.group_ids is not None and len(criteria.group_ids) > 0:
        if context.group_id not in criteria.group_ids:
            return FilterEvaluationResult(
                passed=False,
                reason=f"Excluded: group '{context.group_name}' (ID {context.group_id}) not in selected groups {criteria.group_ids}",
                is_missing_metadata=False,
            )

    # 2. Linked Message Requirement Filter
    if criteria.has_linked_message is not None:
        if criteria.has_linked_message is True and not context.has_linked_message:
            return FilterEvaluationResult(
                passed=False,
                reason="Excluded: document does not have a linked WhatsApp sharing message",
                is_missing_metadata=False,
            )
        if criteria.has_linked_message is False and context.has_linked_message:
            return FilterEvaluationResult(
                passed=False,
                reason="Excluded: document has a linked WhatsApp sharing message (filter requested unlinked documents)",
                is_missing_metadata=False,
            )

    # 3. Sender Filter
    if criteria.senders is not None and len(criteria.senders) > 0:
        target_senders = {s.strip().lower() for s in criteria.senders if s.strip()}
        if target_senders:
            doc_senders = {m.sender.strip().lower() for m in context.all_messages if m.sender and m.sender.strip()}
            if context.sender and context.sender.strip():
                doc_senders.add(context.sender.strip().lower())

            if not doc_senders:
                return FilterEvaluationResult(
                    passed=False,
                    reason=f"Excluded: document has no sender metadata to satisfy sender filter {criteria.senders}",
                    is_missing_metadata=True,
                )

            if not doc_senders.intersection(target_senders):
                sender_display = context.sender or "Unknown"
                return FilterEvaluationResult(
                    passed=False,
                    reason=f"Excluded: sender '{sender_display}' does not match sender filter {criteria.senders}",
                    is_missing_metadata=False,
                )

    # 4. Date Range Filter
    if (criteria.date_from and criteria.date_from.strip()) or (criteria.date_to and criteria.date_to.strip()):
        from_str = _extract_date_str(criteria.date_from)
        to_str = _extract_date_str(criteria.date_to)

        doc_dates: list[str] = []
        for m in context.all_messages:
            d = _extract_date_str(m.sent_at)
            if d:
                doc_dates.append(d)
        if context.sent_at:
            d = _extract_date_str(context.sent_at)
            if d and d not in doc_dates:
                doc_dates.append(d)

        if not doc_dates:
            range_desc = f"[{from_str or '*'} to {to_str or '*'}]"
            return FilterEvaluationResult(
                passed=False,
                reason=f"Excluded: document has no timestamp metadata to evaluate against date range {range_desc}",
                is_missing_metadata=True,
            )

        # Check if at least one linked message falls within the specified date range
        in_range = False
        for d in doc_dates:
            after_from = (from_str is None) or (d >= from_str)
            before_to = (to_str is None) or (d <= to_str)
            if after_from and before_to:
                in_range = True
                break

        if not in_range:
            range_desc = f"[{from_str or '*'} to {to_str or '*'}]"
            return FilterEvaluationResult(
                passed=False,
                reason=f"Excluded: document timestamp '{doc_dates[0]}' is outside requested range {range_desc}",
                is_missing_metadata=False,
            )

    # 5. Message Text Matching Filter
    if criteria.message_query and criteria.message_query.strip():
        q_lower = criteria.message_query.strip().lower()

        # Gather all text associated with WhatsApp messages for this document
        msg_texts: list[str] = []
        for m in context.all_messages:
            if m.message_text:
                msg_texts.append(m.message_text.lower())
            if m.surrounding_context:
                msg_texts.append(m.surrounding_context.lower())

        if not msg_texts:
            return FilterEvaluationResult(
                passed=False,
                reason=f"Excluded: document has no WhatsApp message text to match query '{criteria.message_query}'",
                is_missing_metadata=True,
            )

        matched = any(q_lower in t for t in msg_texts)
        if not matched:
            return FilterEvaluationResult(
                passed=False,
                reason=f"Excluded: WhatsApp message text does not contain query '{criteria.message_query}'",
                is_missing_metadata=False,
            )

    return FilterEvaluationResult(passed=True, reason="Passed all active filters")


def filter_ranked_documents(
    documents: list[Any],
    criteria: WhatsAppFilterCriteria,
    context_map: dict[int, DocumentWhatsAppContext] | None = None,
    db_path: Path | str | None = None,
) -> tuple[list[Any], list[tuple[Any, FilterEvaluationResult]]]:
    """Apply WhatsApp context filters to a list of RankedDocument objects.

    Args:
        documents: List of RankedDocument instances.
        criteria: WhatsAppFilterCriteria containing active constraints.
        context_map: Optional pre-fetched DocumentWhatsAppContext map.
        db_path: Optional custom database path.

    Returns:
        tuple of (surviving_documents, excluded_document_records)
    """
    if not documents:
        return [], []

    if not criteria.is_active():
        return list(documents), []

    # Ensure context is available
    if context_map is None:
        file_ids = [int(getattr(d, "file_id", 0)) for d in documents]
        context_map = get_whatsapp_context_for_documents(file_ids, db_path=db_path)

    surviving: list[Any] = []
    excluded: list[tuple[Any, FilterEvaluationResult]] = []

    for doc in documents:
        fid = int(getattr(doc, "file_id", 0))
        ctx = context_map.get(fid)
        if ctx is None:
            ctx = DocumentWhatsAppContext(
                file_id=fid,
                filename=str(getattr(doc, "filename", f"file_{fid}")),
                group_id=int(getattr(doc, "group_id", 0)),
                group_name=str(getattr(doc, "group_name", "Unknown Group")),
                has_linked_message=False,
            )

        eval_res = evaluate_document_filter(ctx, criteria)
        if eval_res.passed:
            surviving.append(doc)
        else:
            excluded.append((doc, eval_res))

    return surviving, excluded


# ---------------------------------------------------------------------------
# Filter Option Discovery Helpers
# ---------------------------------------------------------------------------

def get_available_senders(
    group_id: int | None = None,
    db_path: Path | str | None = None,
) -> list[str]:
    """Retrieve distinct non-empty senders discovered from the database."""
    return get_distinct_senders(group_id=group_id, db_path=db_path)


def get_available_date_range(
    group_id: int | None = None,
    db_path: Path | str | None = None,
) -> tuple[str | None, str | None]:
    """Retrieve earliest and latest message timestamps discovered from the database."""
    return get_date_bounds(group_id=group_id, db_path=db_path)
