"""WhatsApp chat log parsing module for StudyRAG.

Supports Android and iOS exported chat formats, multiline messages,
attachment detection, ambiguous date resolution, and system message filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import re
from typing import Any

logger = logging.getLogger(__name__)

# Regular expressions for message headers
# Android format: "12/03/24, 10:15 pm - Sender: Message text"
# or "12/03/2024, 10:15:32 PM - Sender: Message text"
# or "12/03/24, 22:15 - Sender: Message text"
ANDROID_HEADER_RE = re.compile(
    r"^(\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp]\.?[Mm]\.?)?)\s+-\s+(.*)$"
)

# iOS format: "[12/03/24, 10:15:32 PM] Sender: Message text"
# or "[12/03/2024, 22:15] Sender: Message text"
IOS_HEADER_RE = re.compile(
    r"^\[(\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp]\.?[Mm]\.?)?)\]\s+(.*)$"
)

# Attachment patterns
# 1. "<attached: filename.ext>"
ATTACHED_BRACKET_RE = re.compile(r"<attached:\s*([^>]+)>", re.IGNORECASE)
# 2. "filename.ext (file attached)"
ATTACHED_PAREN_RE = re.compile(
    r"([\w\-. ]+\.[a-zA-Z0-9]+)\s*\(file attached\)", re.IGNORECASE
)
# 3. Omitted media markers (e.g. "<Media omitted>", "• document omitted", "image omitted")
OMITTED_MEDIA_RE = re.compile(
    r"<\s*media omitted\s*>|•?\s*(?:document|image|video|audio|file|media)\s+omitted",
    re.IGNORECASE,
)

# Characters to strip during Unicode cleanup
STRIP_UNICODE_CHARS = (
    "\u200e",  # Left-to-Right Mark
    "\u200f",  # Right-to-Left Mark
    "\u202a",  # LRE
    "\u202b",  # RLE
    "\u202c",  # PDF
    "\u202d",  # LRO
    "\u202e",  # RLO
    "\ufeff",  # Byte Order Mark (BOM)
    "\u200b",  # Zero-width space
)


@dataclass
class ChatMessage:
    """Structured representation of a parsed WhatsApp message."""

    raw_date: str
    raw_time: str
    sent_at: str | None = None  # Normalized ISO-8601 string (e.g. 2024-03-12T22:15:00)
    datetime_obj: datetime | None = None
    sender: str | None = None
    text: str = ""
    file: str | None = None  # Extracted attachment filename if present
    has_attachment: bool = False  # True if message indicates an attachment was sent
    is_system: bool = False
    is_ambiguous_date: bool = False


def clean_unicode(text: str) -> str:
    """Remove invisible direction marks, BOMs, and normalize narrow spaces."""
    for ch in STRIP_UNICODE_CHARS:
        text = text.replace(ch, "")
    # Normalize narrow no-break space to standard space
    return text.replace("\u202f", " ")


def extract_attachment_info(content: str) -> tuple[str | None, str]:
    """Extract attachment filename and cleaned message body/caption from message content.

    Returns:
        (attachment_filename, remaining_caption_or_text)
    """
    filename: str | None = None
    caption = content

    # Check "<attached: filename>"
    match_bracket = ATTACHED_BRACKET_RE.search(content)
    if match_bracket:
        filename = match_bracket.group(1).strip()
        # Remove attachment marker to leave caption
        caption = ATTACHED_BRACKET_RE.sub("", content).strip()

    # Check "filename (file attached)"
    if not filename:
        match_paren = ATTACHED_PAREN_RE.search(content)
        if match_paren:
            filename = match_paren.group(1).strip()
            caption = ATTACHED_PAREN_RE.sub("", content).strip()

    return filename, caption


def detect_chat_date_format(
    date_strings: list[str],
    default_format: str = "dmy",
) -> tuple[str, bool]:
    """Detect whether dates in a chat are DMY or MDY based on unambiguous days (> 12).

    Args:
        date_strings: List of raw date strings from the chat log.
        default_format: Fallback format ("dmy" or "mdy") if dates are ambiguous.

    Returns:
        tuple of (detected_format, is_ambiguous)
    """
    has_dmy = False
    has_mdy = False

    for date_str in date_strings:
        parts = re.split(r"[/.-]", date_str.strip())
        if len(parts) != 3:
            continue

        p1_str, p2_str, p3_str = parts[0], parts[1], parts[2]
        if not (p1_str.isdigit() and p2_str.isdigit() and p3_str.isdigit()):
            continue

        p1, p2, p3 = int(p1_str), int(p2_str), int(p3_str)

        # If Year is first: YYYY-MM-DD
        if p1 >= 1000:
            return "ymd", False

        # First number > 12 -> must be Day (DMY)
        if p1 > 12 and p2 <= 12:
            has_dmy = True
        # Second number > 12 -> must be Day (MDY)
        elif p2 > 12 and p1 <= 12:
            has_mdy = True

    if has_dmy and not has_mdy:
        return "dmy", False
    if has_mdy and not has_dmy:
        return "mdy", False

    # Ambiguous or conflicting
    return default_format.lower(), True


def parse_timestamp(
    date_str: str,
    time_str: str,
    date_format: str = "dmy",
) -> tuple[datetime | None, str | None]:
    """Parse raw date and time strings into a datetime object and ISO string."""
    try:
        parts = re.split(r"[/.-]", date_str.strip())
        if len(parts) != 3:
            return None, None

        n1, n2, n3 = int(parts[0]), int(parts[1]), int(parts[2])

        if date_format == "ymd" or n1 >= 1000:
            year, month, day = n1, n2, n3
        elif date_format == "mdy":
            month, day, year = n1, n2, n3
        else:  # dmy default
            day, month, year = n1, n2, n3

        # Normalize 2-digit years
        if year < 100:
            year += 2000

        # Parse time string: e.g. "10:15", "10:15:32", "10:15 pm", "10:15:32 PM"
        time_clean = time_str.strip().lower()
        is_pm = "pm" in time_clean or "p.m." in time_clean
        is_am = "am" in time_clean or "a.m." in time_clean

        # Remove AM/PM text to isolate numbers
        time_digits = re.sub(r"[a-z.\s]", "", time_clean)
        time_parts = [int(p) for p in time_digits.split(":") if p.isdigit()]

        if not time_parts:
            return None, None

        hour = time_parts[0]
        minute = time_parts[1] if len(time_parts) > 1 else 0
        second = time_parts[2] if len(time_parts) > 2 else 0

        # Convert 12-hour format to 24-hour format
        if is_pm and hour < 12:
            hour += 12
        elif is_am and hour == 12:
            hour = 0

        dt = datetime(year, month, day, hour, minute, second)
        return dt, dt.isoformat()
    except (ValueError, IndexError):
        return None, None


def parse_chat(
    text: str,
    default_date_format: str = "dmy",
) -> list[ChatMessage]:
    """Parse exported WhatsApp chat text into structured ChatMessage objects.

    Supports Android and iOS formats, multiline messages, system messages,
    attachments, and ambiguous date resolution.

    Args:
        text: Raw text of exported WhatsApp chat.
        default_date_format: Fallback date format ("dmy" or "mdy") if ambiguous.

    Returns:
        List of structured ChatMessage instances.
    """
    cleaned_text = clean_unicode(text)
    lines = cleaned_text.splitlines()

    # Pass 1: Collect raw date strings to determine chat-wide date format
    raw_dates: list[str] = []
    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        android_match = ANDROID_HEADER_RE.match(line_s)
        if android_match:
            raw_dates.append(android_match.group(1))
            continue
        ios_match = IOS_HEADER_RE.match(line_s)
        if ios_match:
            raw_dates.append(ios_match.group(1))

    detected_format, is_ambiguous = detect_chat_date_format(
        raw_dates, default_format=default_date_format
    )

    # Pass 2: Parse messages and handle multiline continuations
    messages: list[ChatMessage] = []
    current_msg: ChatMessage | None = None

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            # Empty line inside an active message belongs to the message body
            if current_msg is not None:
                current_msg.text += "\n"
            continue

        match = ANDROID_HEADER_RE.match(line) or IOS_HEADER_RE.match(line)
        if match:
            # Finalize previous message attachment check
            if current_msg is not None:
                _finalize_message(current_msg)
                messages.append(current_msg)

            raw_d, raw_t, remainder = match.group(1), match.group(2), match.group(3)
            dt_obj, iso_ts = parse_timestamp(raw_d, raw_t, date_format=detected_format)

            # Check if remainder has "Sender: Message"
            # Format: everything before the first colon is the sender
            if ": " in remainder:
                sender_part, msg_text = remainder.split(": ", 1)
                sender = sender_part.strip()
                text_content = msg_text
                is_system = False
            else:
                # System message (e.g. "Messages and calls are end-to-end encrypted")
                sender = None
                text_content = remainder
                is_system = True

            current_msg = ChatMessage(
                raw_date=raw_d,
                raw_time=raw_t,
                sent_at=iso_ts,
                datetime_obj=dt_obj,
                sender=sender,
                text=text_content,
                is_system=is_system,
                is_ambiguous_date=is_ambiguous,
            )
        else:
            # Continuation line for the current message
            if current_msg is not None:
                if current_msg.text:
                    current_msg.text += "\n" + line
                else:
                    current_msg.text = line
            else:
                # Leading non-message lines (e.g. file headers) ignored
                continue

    # Finalize last message
    if current_msg is not None:
        _finalize_message(current_msg)
        messages.append(current_msg)

    return messages


def _finalize_message(msg: ChatMessage) -> None:
    """Extract attachment filename and clean message text."""
    filename, caption = extract_attachment_info(msg.text)
    if filename:
        msg.file = filename
        msg.has_attachment = True
    elif OMITTED_MEDIA_RE.search(msg.text):
        msg.has_attachment = True
        msg.file = None


def find_chat_file(directory: Path) -> Path | None:
    """Identify the primary WhatsApp chat file in an extracted directory.

    Rules:
    1. Prefer the first .txt file whose filename contains 'chat' (case-insensitive).
    2. Fall back to any other .txt file in the directory.
    3. Never select .md files as chat logs.

    Args:
        directory: Directory containing extracted export files.

    Returns:
        Path to chat file if found, otherwise None.
    """
    if not directory.exists() or not directory.is_dir():
        return None

    txt_files = sorted(
        [p for p in directory.glob("*.txt") if p.is_file()],
        key=lambda p: p.name.lower(),
    )

    # 1. Prefer file containing 'chat'
    for p in txt_files:
        if "chat" in p.name.lower():
            return p

    # 2. Fall back to first available .txt file
    if txt_files:
        return txt_files[0]

    return None
