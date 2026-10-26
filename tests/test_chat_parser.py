"""Tests for WhatsApp chat parser module (core/chat_parser.py)."""

from datetime import datetime
from pathlib import Path
import pytest

from core.chat_parser import (
    clean_unicode,
    detect_chat_date_format,
    extract_attachment_info,
    find_chat_file,
    parse_chat,
    parse_timestamp,
)


def test_clean_unicode():
    """Verify stripping of invisible marks, BOM, and narrow spaces."""
    raw = "\u200eHello\u200f \ufeffWorld\u202fPM\u200b!"
    cleaned = clean_unicode(raw)
    assert cleaned == "Hello World PM!"


def test_android_format_parsing():
    """Verify parsing of Android format messages."""
    chat = (
        "12/03/24, 10:15 pm - Alice: Here is the lecture slides\n"
        "12/03/24, 10:16 pm - Bob: Thanks Alice!\n"
    )
    messages = parse_chat(chat)
    assert len(messages) == 2

    assert messages[0].sender == "Alice"
    assert messages[0].text == "Here is the lecture slides"
    assert messages[0].sent_at == "2024-03-12T22:15:00"
    assert messages[0].is_system is False

    assert messages[1].sender == "Bob"
    assert messages[1].text == "Thanks Alice!"
    assert messages[1].sent_at == "2024-03-12T22:16:00"


def test_ios_format_parsing():
    """Verify parsing of iOS format messages."""
    chat = (
        "[12/03/2024, 10:15:32 AM] Dr. Smith: Assignment 1 released.\n"
        "[12/03/2024, 10:16:00 AM] Charlie: Received, thank you.\n"
    )
    messages = parse_chat(chat)
    assert len(messages) == 2

    assert messages[0].sender == "Dr. Smith"
    assert messages[0].text == "Assignment 1 released."
    assert messages[0].sent_at == "2024-03-12T10:15:32"

    assert messages[1].sender == "Charlie"
    assert messages[1].text == "Received, thank you."
    assert messages[1].sent_at == "2024-03-12T10:16:00"


def test_timestamps_with_and_without_seconds():
    """Verify 12-hour, 24-hour, with and without seconds parsing."""
    dt1, iso1 = parse_timestamp("12/03/2024", "10:15 pm")
    assert iso1 == "2024-03-12T22:15:00"

    dt2, iso2 = parse_timestamp("12/03/2024", "10:15:45 am")
    assert iso2 == "2024-03-12T10:15:45"

    dt3, iso3 = parse_timestamp("12/03/2024", "23:59:01")
    assert iso3 == "2024-03-12T23:59:01"

    dt4, iso4 = parse_timestamp("12/03/2024", "12:00 am")
    assert iso4 == "2024-03-12T00:00:00"


def test_multiline_messages():
    """Continuation lines without timestamps must append to previous message."""
    chat = (
        "12/03/24, 10:15 pm - Alice: Line 1 of message\n"
        "Line 2 continuation\n"
        "Line 3 continuation with details\n"
        "12/03/24, 10:18 pm - Bob: Acknowledged.\n"
    )
    messages = parse_chat(chat)
    assert len(messages) == 2
    assert messages[0].sender == "Alice"
    assert (
        messages[0].text
        == "Line 1 of message\nLine 2 continuation\nLine 3 continuation with details"
    )
    assert messages[1].sender == "Bob"
    assert messages[1].text == "Acknowledged."


def test_attachment_patterns():
    """Verify detection of <attached: file> and 'file (file attached)' patterns."""
    # Pattern 1: <attached: filename.pdf>
    text1 = "Here are notes <attached: chapter1.pdf> Please study"
    f1, cap1 = extract_attachment_info(text1)
    assert f1 == "chapter1.pdf"
    assert cap1 == "Here are notes  Please study"

    # Pattern 2: filename (file attached)
    text2 = "midterm_prep.docx (file attached) Bring a calculator"
    f2, cap2 = extract_attachment_info(text2)
    assert f2 == "midterm_prep.docx"
    assert cap2 == "Bring a calculator"

    # Through parse_chat
    chat = (
        "12/03/24, 10:15 pm - Alice: <attached: syllabus.pdf>\n"
        "12/03/24, 10:16 pm - Bob: formulas.pdf (file attached)\n"
    )
    msgs = parse_chat(chat)
    assert msgs[0].file == "syllabus.pdf"
    assert msgs[1].file == "formulas.pdf"


def test_date_format_detection_unambiguous():
    """Unambiguous dates (>12) correctly detect DMY or MDY."""
    # Day > 12 -> DMY
    fmt_dmy, amb1 = detect_chat_date_format(["25/03/2024", "05/03/2024"])
    assert fmt_dmy == "dmy"
    assert amb1 is False

    # Month > 12 -> impossible, so p2 > 12 means Month/Day/Year (MDY)
    fmt_mdy, amb2 = detect_chat_date_format(["03/25/2024", "03/05/2024"])
    assert fmt_mdy == "mdy"
    assert amb2 is False


def test_date_format_detection_ambiguous_fallback():
    """Ambiguous dates (<= 12) fall back to configurable default."""
    fmt, amb = detect_chat_date_format(["05/06/2024", "01/02/2024"], default_format="dmy")
    assert fmt == "dmy"
    assert amb is True

    fmt2, amb2 = detect_chat_date_format(["05/06/2024"], default_format="mdy")
    assert fmt2 == "mdy"
    assert amb2 is True


def test_system_messages_and_unusual_senders():
    """System messages (no ': ') and unusual sender names are handled safely."""
    chat = (
        "12/03/24, 10:00 pm - Messages and calls are end-to-end encrypted.\n"
        "12/03/24, 10:01 pm - Alice created group \"Study Group\"\n"
        "12/03/24, 10:02 pm - +91 98765 43210: Hello all\n"
        "12/03/24, 10:03 pm - Prof. Smith 🎓 - CS Dept: Welcome everyone!\n"
    )
    msgs = parse_chat(chat)
    assert len(msgs) == 4

    assert msgs[0].is_system is True
    assert msgs[0].sender is None
    assert "end-to-end encrypted" in msgs[0].text

    assert msgs[1].is_system is True
    assert msgs[1].sender is None

    assert msgs[2].is_system is False
    assert msgs[2].sender == "+91 98765 43210"
    assert msgs[2].text == "Hello all"

    assert msgs[3].is_system is False
    assert msgs[3].sender == "Prof. Smith 🎓 - CS Dept"
    assert msgs[3].text == "Welcome everyone!"


def test_malformed_lines_without_crashing():
    """Random malformed text at the beginning or middle does not crash parser."""
    chat = (
        "=== Chat Export Header Line ===\n"
        "Random notes without timestamp\n"
        "12/03/24, 10:15 pm - Alice: Valid message\n"
        "!!! Corrupt line that does not match anything\n"
        "12/03/24, 10:16 pm - Bob: Another valid message\n"
    )
    msgs = parse_chat(chat)
    assert len(msgs) == 2
    assert msgs[0].sender == "Alice"
    assert "Corrupt line" in msgs[0].text  # Appended as multiline to Alice
    assert msgs[1].sender == "Bob"


def test_find_chat_file_rules(tmp_path: Path):
    """Verify chat file discovery: prefer *chat*.txt, fall back to *.txt, ignore *.md."""
    # 1. Directory with chat file and other files
    (tmp_path / "notes.md").write_text("# Markdown", encoding="utf-8")
    (tmp_path / "random.txt").write_text("random", encoding="utf-8")
    chat_file = tmp_path / "_chat.txt"
    chat_file.write_text("chat content", encoding="utf-8")

    found = find_chat_file(tmp_path)
    assert found == chat_file

    # 2. When no file with 'chat' in name exists, falls back to other .txt
    chat_file.unlink()
    found_fallback = find_chat_file(tmp_path)
    assert found_fallback == tmp_path / "random.txt"

    # 3. When only .md exists, returns None (never returns .md)
    (tmp_path / "random.txt").unlink()
    assert find_chat_file(tmp_path) is None
