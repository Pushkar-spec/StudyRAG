"""SQLite database foundation and schema management for StudyRAG.

Provides idempotent schema initialization, index creation, foreign key enforcement,
FTS5 full-text search virtual tables, and safe transaction context management.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
import sqlite3
from typing import Generator

from core.config import get_db_path

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for StudyRAG database operations."""


class FTS5NotSupportedError(DatabaseError):
    """Raised when SQLite FTS5 extension is unavailable in the environment."""


__all__ = [
    "DatabaseError",
    "FTS5NotSupportedError",
    "check_fts5_support",
    "get_connection",
    "db_session",
    "init_db",
    "get_existing_tables",
    "get_or_create_group",
    "get_file_by_sha256",
    "insert_file",
    "register_sender",
    "get_sender",
    "get_all_groups",
    "get_group_files",
    "get_file_by_id",
    "update_file_metadata",
    "upsert_attachment_ref",
    "get_attachment_ref",
    "get_attachment_refs",
    "get_attachment_ref_counts",
    "update_attachment_ref_resolution",
    "get_file_chunks",
    "get_processing_status_counts",
    "update_file_processing_status",
    "sanitize_fts5_query",
    "search_chunks_fts",
    "get_pending_chunks_for_indexing",
    "record_chunks_indexed_batch",
    "delete_file_vector_state",
    "update_file_indexing_status",
    "get_vector_index_counts",
    "get_chunk_with_metadata",
]


SCHEMA_STATEMENTS = [
    # 1. Groups table: represents an exported WhatsApp study group
    """
    CREATE TABLE IF NOT EXISTS groups (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    );
    """,

    # 2. Files table: tracks study files discovered within groups
    """
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
        sha256 TEXT NOT NULL,
        filename TEXT NOT NULL,
        path TEXT NOT NULL,
        ext TEXT NOT NULL,
        sender TEXT,
        sent_at TEXT,
        caption TEXT,
        status TEXT NOT NULL DEFAULT 'new',
        error_message TEXT,
        chunk_count INTEGER DEFAULT 0,
        processed_at TEXT,
        indexed_chunk_count INTEGER DEFAULT 0,
        is_indexed INTEGER DEFAULT 0,
        UNIQUE(group_id, sha256)
    );
    """,

    # 3. Chunks table: text chunks extracted from files
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        page INTEGER,
        text TEXT NOT NULL,
        location TEXT,
        chunk_index INTEGER
    );
    """,

    # 4. Senders table: tracks senders and professor status per group
    """
    CREATE TABLE IF NOT EXISTS senders (
        group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        is_professor INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (group_id, name)
    );
    """,

    # 5. OCR Cache table: caches OCR results per file page with confidence metric
    """
    CREATE TABLE IF NOT EXISTS ocr_cache (
        sha256 TEXT NOT NULL,
        page INTEGER NOT NULL,
        text TEXT NOT NULL,
        confidence REAL,
        PRIMARY KEY (sha256, page)
    );
    """,

    # 6. LLM Cache table: caches prompt/grading results to prevent redundant calls
    """
    CREATE TABLE IF NOT EXISTS llm_cache (
        key TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        value TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
    );
    """,

    # 7. Usage table: tracks daily API token and call usage
    """
    CREATE TABLE IF NOT EXISTS usage (
        day TEXT PRIMARY KEY,
        calls INTEGER NOT NULL DEFAULT 0,
        prompt_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        cache_hits INTEGER NOT NULL DEFAULT 0
    );
    """,

    # 8. Attachment References table: tracks referenced attachments and recovery status
    """
    CREATE TABLE IF NOT EXISTS attachment_refs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
        file_id INTEGER REFERENCES files(id) ON DELETE SET NULL,
        original_filename TEXT,
        normalized_filename TEXT,
        sender TEXT,
        sent_at TEXT,
        caption TEXT,
        context TEXT,
        status TEXT NOT NULL,
        resolution_method TEXT,
        resolution_note TEXT,
        dedup_key TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
        updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
        UNIQUE(group_id, dedup_key)
    );
    """,

    # 9. Vector index state table: tracks chunk embedding and sync state
    """
    CREATE TABLE IF NOT EXISTS vector_index_state (
        chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
        file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        embed_model TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        indexed_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
    );
    """,
]

INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_files_group_status ON files(group_id, status);",
    "CREATE INDEX IF NOT EXISTS idx_files_sha256 ON files(sha256);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_file_page ON chunks(file_id, page);",
    "CREATE INDEX IF NOT EXISTS idx_ocr_cache_sha256 ON ocr_cache(sha256);",
    "CREATE INDEX IF NOT EXISTS idx_llm_cache_kind ON llm_cache(kind);",
    "CREATE INDEX IF NOT EXISTS idx_attachment_refs_group_status ON attachment_refs(group_id, status);",
    "CREATE INDEX IF NOT EXISTS idx_attachment_refs_norm_filename ON attachment_refs(group_id, normalized_filename);",
    "CREATE INDEX IF NOT EXISTS idx_attachment_refs_file_id ON attachment_refs(file_id);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_file_chunk_index ON chunks(file_id, chunk_index);",
    "CREATE INDEX IF NOT EXISTS idx_vector_index_file ON vector_index_state(file_id);",
    "CREATE INDEX IF NOT EXISTS idx_vector_index_model ON vector_index_state(embed_model);",
]

# FTS5 Virtual Table & Synchronization Triggers
FTS_STATEMENTS = [
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        id UNINDEXED,
        text
    );
    """,
    # Trigger to sync chunk insertions into FTS index
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(id, text) VALUES (new.id, new.text);
    END;
    """,
    # Trigger to sync chunk deletions from FTS index
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        DELETE FROM chunks_fts WHERE id = old.id;
    END;
    """,
    # Trigger to sync chunk updates in FTS index
    """
    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
        DELETE FROM chunks_fts WHERE id = old.id;
        INSERT INTO chunks_fts(id, text) VALUES (new.id, new.text);
    END;
    """,
]


def check_fts5_support(conn: sqlite3.Connection | None = None) -> bool:
    """Verify if SQLite FTS5 extension is available.

    Args:
        conn: Optional existing connection. If None, a temporary in-memory connection is used.

    Returns:
        True if FTS5 is supported.

    Raises:
        FTS5NotSupportedError: If FTS5 module cannot be initialized.
    """
    close_after = False
    if conn is None:
        conn = sqlite3.connect(":memory:")
        close_after = True

    try:
        cur = conn.cursor()
        cur.execute("CREATE VIRTUAL TABLE temp._fts5_check USING fts5(x);")
        cur.execute("DROP TABLE temp._fts5_check;")
        return True
    except sqlite3.OperationalError as err:
        if "no such module: fts5" in str(err).lower():
            raise FTS5NotSupportedError(
                "SQLite FTS5 extension is not available in the current Python SQLite build. "
                "StudyRAG requires FTS5 for full-text document search."
            ) from err
        raise DatabaseError(f"Error checking FTS5 support: {err}") from err
    finally:
        if close_after:
            conn.close()


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Create a configured SQLite connection with foreign key constraints enabled.

    Args:
        db_path: Path to database file. If None, uses default get_db_path().

    Returns:
        sqlite3.Connection with foreign keys and row factory enabled.
    """
    target_path = Path(db_path).resolve() if db_path else get_db_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(target_path),
        timeout=10.0,
        detect_types=sqlite3.PARSE_DECLTYPES,
    )
    conn.row_factory = sqlite3.Row
    # Enforce foreign key constraints on every connection
    conn.execute("PRAGMA foreign_keys = ON;")
    # Set busy timeout for concurrent access safety
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


@contextmanager
def db_session(db_path: Path | str | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager providing an active database connection within a transaction.

    Commits on normal exit, rolls back on exception, and always closes the connection.
    """
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str | None = None) -> Path:
    """Idempotently initialize StudyRAG database schema, indexes, and FTS5 table.

    Calling this multiple times is safe and preserves existing data.

    Args:
        db_path: Path to target database. Defaults to get_db_path().

    Returns:
        Path to the initialized database.

    Raises:
        FTS5NotSupportedError: If FTS5 is not available.
        DatabaseError: If schema creation fails.
    """
    target_path = Path(db_path).resolve() if db_path else get_db_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with get_connection(target_path) as conn:
        # Check FTS5 availability first
        check_fts5_support(conn)

        try:
            cur = conn.cursor()
            # 1. Base tables
            for stmt in SCHEMA_STATEMENTS:
                cur.execute(stmt)

            # 2. Safe additive migrations for existing databases
            cur.execute("PRAGMA table_info(files);")
            file_cols = {row["name"] for row in cur.fetchall()}
            if "sender" not in file_cols:
                cur.execute("ALTER TABLE files ADD COLUMN sender TEXT;")
            if "sent_at" not in file_cols:
                cur.execute("ALTER TABLE files ADD COLUMN sent_at TEXT;")
            if "caption" not in file_cols:
                cur.execute("ALTER TABLE files ADD COLUMN caption TEXT;")
            if "error_message" not in file_cols:
                cur.execute("ALTER TABLE files ADD COLUMN error_message TEXT;")
            if "chunk_count" not in file_cols:
                cur.execute("ALTER TABLE files ADD COLUMN chunk_count INTEGER DEFAULT 0;")
            if "processed_at" not in file_cols:
                cur.execute("ALTER TABLE files ADD COLUMN processed_at TEXT;")
            if "indexed_chunk_count" not in file_cols:
                cur.execute("ALTER TABLE files ADD COLUMN indexed_chunk_count INTEGER DEFAULT 0;")
            if "is_indexed" not in file_cols:
                cur.execute("ALTER TABLE files ADD COLUMN is_indexed INTEGER DEFAULT 0;")

            cur.execute("PRAGMA table_info(chunks);")
            chunk_cols = {row["name"] for row in cur.fetchall()}
            if "location" not in chunk_cols:
                cur.execute("ALTER TABLE chunks ADD COLUMN location TEXT;")
            if "chunk_index" not in chunk_cols:
                cur.execute("ALTER TABLE chunks ADD COLUMN chunk_index INTEGER;")

            # 3. Deterministic backfill for legacy chunk rows
            cur.execute(
                """
                UPDATE chunks
                SET chunk_index = (
                    SELECT rn - 1
                    FROM (
                        SELECT id, ROW_NUMBER() OVER (
                            PARTITION BY file_id 
                            ORDER BY 
                                CASE WHEN page IS NOT NULL AND page > 0 THEN page ELSE 999999 END ASC, 
                                id ASC, 
                                rowid ASC
                        ) AS rn
                        FROM chunks
                    ) AS ranked
                    WHERE ranked.id = chunks.id
                )
                WHERE chunk_index IS NULL;
                """
            )
            cur.execute(
                """
                UPDATE chunks
                SET location = 'Page ' || page
                WHERE (location IS NULL OR location = '') AND page IS NOT NULL AND page > 0;
                """
            )
            cur.execute(
                """
                UPDATE files
                SET chunk_count = (
                    SELECT COUNT(*) FROM chunks WHERE chunks.file_id = files.id
                )
                WHERE (chunk_count IS NULL OR chunk_count = 0) AND status = 'processed';
                """
            )

            # 4. Dependent indexes (safe now that chunk_index and all columns exist)
            for stmt in INDEX_STATEMENTS:
                cur.execute(stmt)

            # 5. FTS5 virtual table and synchronization triggers
            for stmt in FTS_STATEMENTS:
                cur.execute(stmt)

            # 6. Ensure FTS5 index contains all existing chunks
            cur.execute(
                """
                INSERT INTO chunks_fts (id, text)
                SELECT id, text FROM chunks
                WHERE id NOT IN (SELECT id FROM chunks_fts);
                """
            )

            conn.commit()
            logger.info("Database initialized successfully at %s", target_path)
        except sqlite3.Error as err:
            conn.rollback()
            raise DatabaseError(f"Failed to initialize database schema at {target_path}: {err}") from err

    return target_path


def get_existing_tables(db_path: Path | str | None = None) -> list[str]:
    """Return a list of non-internal tables and virtual tables in the database."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT name FROM sqlite_master 
            WHERE type IN ('table', 'view') 
              AND name NOT LIKE 'sqlite_%'
              AND name NOT LIKE 'chunks_fts_%'
            ORDER BY name;
            """
        )
        return [row["name"] for row in cursor.fetchall()]


def get_or_create_group(name: str, db_path: Path | str | None = None) -> int:
    """Retrieve existing group ID or create a new group record.

    Args:
        name: Name of the study group.
        db_path: Optional custom database path.

    Returns:
        int: The group ID.
    """
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Group name cannot be empty")

    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM groups WHERE name = ?;", (clean_name,))
        row = cursor.fetchone()
        if row:
            return int(row["id"])

        cursor.execute("INSERT INTO groups (name) VALUES (?);", (clean_name,))
        return int(cursor.lastrowid)


def get_file_by_sha256(
    group_id: int,
    sha256: str,
    db_path: Path | str | None = None,
) -> sqlite3.Row | None:
    """Retrieve a file record within a group matching the given SHA-256 hash."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM files WHERE group_id = ? AND sha256 = ?;",
            (group_id, sha256),
        )
        return cursor.fetchone()


def insert_file(
    group_id: int,
    sha256: str,
    filename: str,
    path: str,
    ext: str,
    sender: str | None = None,
    sent_at: str | None = None,
    caption: str | None = None,
    status: str = "new",
    db_path: Path | str | None = None,
) -> int:
    """Insert a new file record into the files table.

    Returns:
        int: The inserted file ID.
    """
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO files (group_id, sha256, filename, path, ext, sender, sent_at, caption, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (group_id, sha256, filename, path, ext, sender, sent_at, caption, status),
        )
        return int(cursor.lastrowid)


def register_sender(
    group_id: int,
    name: str,
    is_professor: int = 0,
    db_path: Path | str | None = None,
) -> None:
    """Register a sender in the senders table.

    Preserves existing is_professor status on conflict.
    """
    clean_name = name.strip()
    if not clean_name:
        return

    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO senders (group_id, name, is_professor)
            VALUES (?, ?, ?)
            ON CONFLICT (group_id, name) DO NOTHING;
            """,
            (group_id, clean_name, is_professor),
        )


def get_sender(
    group_id: int,
    name: str,
    db_path: Path | str | None = None,
) -> sqlite3.Row | None:
    """Retrieve a sender record for a specific group."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM senders WHERE group_id = ? AND name = ?;",
            (group_id, name.strip()),
        )
        return cursor.fetchone()


def get_all_groups(db_path: Path | str | None = None) -> list[sqlite3.Row]:
    """Retrieve all study groups ordered by name."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM groups ORDER BY name ASC;")
        return cursor.fetchall()


def get_group_files(
    group_id: int,
    db_path: Path | str | None = None,
) -> list[sqlite3.Row]:
    """Retrieve all files registered for a specific group."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM files WHERE group_id = ? ORDER BY filename ASC;",
            (group_id,),
        )
        return cursor.fetchall()


def get_file_by_id(
    file_id: int,
    db_path: Path | str | None = None,
) -> sqlite3.Row | None:
    """Retrieve a file record by its primary key ID."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM files WHERE id = ?;", (file_id,))
        return cursor.fetchone()


def update_file_metadata(
    file_id: int,
    sender: str | None = None,
    sent_at: str | None = None,
    caption: str | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Update sender, sent_at, and caption metadata for an existing file."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE files
            SET sender = COALESCE(?, sender),
                sent_at = COALESCE(?, sent_at),
                caption = COALESCE(?, caption)
            WHERE id = ?;
            """,
            (sender, sent_at, caption, file_id),
        )


def upsert_attachment_ref(
    group_id: int,
    dedup_key: str,
    status: str,
    original_filename: str | None = None,
    normalized_filename: str | None = None,
    sender: str | None = None,
    sent_at: str | None = None,
    caption: str | None = None,
    context: str | None = None,
    file_id: int | None = None,
    resolution_method: str | None = None,
    resolution_note: str | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Insert or update an attachment reference in the attachment_refs table.

    Returns:
        int: The reference ID.
    """
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        # Check if record already exists
        cursor.execute(
            "SELECT id, status, file_id, resolution_method FROM attachment_refs WHERE group_id = ? AND dedup_key = ?;",
            (group_id, dedup_key),
        )
        existing = cursor.fetchone()

        if existing:
            ref_id = int(existing["id"])
            # If already resolved manually, preserve manual resolution
            if existing["resolution_method"] == "manual" and resolution_method != "manual":
                return ref_id

            cursor.execute(
                """
                UPDATE attachment_refs
                SET file_id = COALESCE(?, file_id),
                    status = ?,
                    resolution_method = COALESCE(?, resolution_method),
                    resolution_note = COALESCE(?, resolution_note),
                    original_filename = COALESCE(?, original_filename),
                    normalized_filename = COALESCE(?, normalized_filename),
                    sender = COALESCE(?, sender),
                    sent_at = COALESCE(?, sent_at),
                    caption = COALESCE(?, caption),
                    context = COALESCE(?, context),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?;
                """,
                (
                    file_id,
                    status,
                    resolution_method,
                    resolution_note,
                    original_filename,
                    normalized_filename,
                    sender,
                    sent_at,
                    caption,
                    context,
                    ref_id,
                ),
            )
            return ref_id

        # Insert new reference
        cursor.execute(
            """
            INSERT INTO attachment_refs (
                group_id, file_id, original_filename, normalized_filename,
                sender, sent_at, caption, context, status,
                resolution_method, resolution_note, dedup_key
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                group_id,
                file_id,
                original_filename,
                normalized_filename,
                sender,
                sent_at,
                caption,
                context,
                status,
                resolution_method,
                resolution_note,
                dedup_key,
            ),
        )
        return int(cursor.lastrowid)


def get_attachment_ref(
    ref_id: int,
    db_path: Path | str | None = None,
) -> sqlite3.Row | None:
    """Retrieve an attachment reference by ID."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attachment_refs WHERE id = ?;", (ref_id,))
        return cursor.fetchone()


def get_attachment_refs(
    group_id: int | None = None,
    status: str | None = None,
    db_path: Path | str | None = None,
) -> list[sqlite3.Row]:
    """Retrieve attachment references filtered by group and/or status."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM attachment_refs WHERE 1=1"
        params: list[Any] = []

        if group_id is not None:
            query += " AND group_id = ?"
            params.append(group_id)

        if status is not None and status.lower() != "all":
            query += " AND status = ?"
            params.append(status.lower())

        query += " ORDER BY id DESC;"
        cursor.execute(query, tuple(params))
        return cursor.fetchall()


def get_attachment_ref_counts(
    group_id: int | None = None,
    db_path: Path | str | None = None,
) -> dict[str, int]:
    """Retrieve count of attachment references aggregated by status."""
    counts = {
        "missing": 0,
        "available": 0,
        "ambiguous": 0,
        "unsupported": 0,
        "import_failed": 0,
        "total": 0,
    }

    with db_session(db_path) as conn:
        cursor = conn.cursor()
        if group_id is not None:
            cursor.execute(
                "SELECT status, COUNT(*) as cnt FROM attachment_refs WHERE group_id = ? GROUP BY status;",
                (group_id,),
            )
        else:
            cursor.execute(
                "SELECT status, COUNT(*) as cnt FROM attachment_refs GROUP BY status;"
            )

        total = 0
        for row in cursor.fetchall():
            s = row["status"]
            cnt = int(row["cnt"])
            if s in counts:
                counts[s] = cnt
            total += cnt
        counts["total"] = total

    return counts


def update_attachment_ref_resolution(
    ref_id: int,
    file_id: int | None,
    status: str,
    method: str,
    note: str,
    db_path: Path | str | None = None,
) -> None:
    """Update status and resolution information for an attachment reference."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE attachment_refs
            SET file_id = ?,
                status = ?,
                resolution_method = ?,
                resolution_note = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?;
            """,
            (file_id, status, method, note, ref_id),
        )


def get_file_chunks(file_id: int | str, db_path: Path | str | None = None) -> list[sqlite3.Row]:
    """Retrieve all text chunks belonging to a specific file ordered by chunk index."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, file_id, page, text, location, chunk_index
            FROM chunks
            WHERE file_id = ?
            ORDER BY COALESCE(chunk_index, 0) ASC, id ASC;
            """,
            (int(file_id),),
        )
        return cursor.fetchall()


def get_processing_status_counts(
    group_id: int | None = None,
    db_path: Path | str | None = None,
) -> dict[str, int]:
    """Return aggregated file processing status counts for a group or across all groups."""
    counts = {
        "new": 0,
        "processed": 0,
        "empty": 0,
        "unsupported": 0,
        "failed": 0,
        "total": 0,
    }
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        if group_id is not None:
            cursor.execute(
                "SELECT status, COUNT(*) as cnt FROM files WHERE group_id = ? GROUP BY status;",
                (group_id,),
            )
        else:
            cursor.execute("SELECT status, COUNT(*) as cnt FROM files GROUP BY status;")

        total = 0
        for row in cursor.fetchall():
            s = row["status"]
            cnt = int(row["cnt"])
            if s in counts:
                counts[s] = cnt
            total += cnt
        counts["total"] = total

    return counts


def update_file_processing_status(
    file_id: int,
    status: str,
    chunk_count: int = 0,
    error_message: str | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Update file processing status, chunk count, and optional error message."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE files
            SET status = ?,
                chunk_count = ?,
                error_message = ?,
                processed_at = CURRENT_TIMESTAMP
            WHERE id = ?;
            """,
            (status, chunk_count, error_message, file_id),
        )


def sanitize_fts5_query(query: str) -> str:
    """Escape and format a search query safely for SQLite FTS5 matching."""
    q = query.strip()
    if not q:
        return '""'
    # If already enclosed in double quotes, preserve phrase matching
    if q.startswith('"') and q.endswith('"') and len(q) > 1:
        inner = q[1:-1].replace('"', '""')
        return f'"{inner}"'
    # If query contains special FTS5 operators/punctuation, wrap in quotes as a phrase
    if any(c in q for c in ('-', ':', '*', '(', ')', '^', '+', '/')):
        escaped = q.replace('"', '""')
        return f'"{escaped}"'
    return q


def search_chunks_fts(
    query: str,
    group_id: int | None = None,
    limit: int = 50,
    db_path: Path | str | None = None,
) -> list[sqlite3.Row]:
    """Execute FTS5 search across chunks with optional study group isolation."""
    safe_query = sanitize_fts5_query(query)
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        if group_id is not None:
            cursor.execute(
                """
                SELECT c.id, c.file_id, c.page, c.text, c.location, c.chunk_index,
                       f.filename, f.group_id, f.sender
                FROM chunks_fts fts
                JOIN chunks c ON fts.id = c.id
                JOIN files f ON c.file_id = f.id
                WHERE chunks_fts MATCH ? AND f.group_id = ?
                LIMIT ?;
                """,
                (safe_query, group_id, limit),
            )
        else:
            cursor.execute(
                """
                SELECT c.id, c.file_id, c.page, c.text, c.location, c.chunk_index,
                       f.filename, f.group_id, f.sender
                FROM chunks_fts fts
                JOIN chunks c ON fts.id = c.id
                JOIN files f ON c.file_id = f.id
                WHERE chunks_fts MATCH ?
                LIMIT ?;
                """,
                (safe_query, limit),
            )
        return cursor.fetchall()


def get_pending_chunks_for_indexing(
    group_id: int | None = None,
    embed_model: str = "BAAI/bge-small-en-v1.5",
    db_path: Path | str | None = None,
) -> list[sqlite3.Row]:
    """Retrieve chunks belonging to processed files that have not yet been indexed with embed_model."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        query = """
            SELECT c.id, c.file_id, c.page, c.text, c.location, c.chunk_index,
                   f.group_id, f.filename, f.sender, v.content_hash as current_indexed_hash
            FROM chunks c
            JOIN files f ON c.file_id = f.id
            LEFT JOIN vector_index_state v ON c.id = v.chunk_id AND v.embed_model = ?
            WHERE f.status = 'processed'
        """
        params: list[Any] = [embed_model]
        if group_id is not None:
            query += " AND f.group_id = ?"
            params.append(group_id)

        query += " ORDER BY c.file_id ASC, COALESCE(c.chunk_index, 0) ASC;"
        cursor.execute(query, tuple(params))
        return cursor.fetchall()


def record_chunks_indexed_batch(
    records: list[tuple[str, int, str, str]],
    db_path: Path | str | None = None,
) -> None:
    """Batch-upsert vector index state records for indexed chunks.

    Args:
        records: List of tuples (chunk_id, file_id, embed_model, content_hash)
        db_path: Optional custom database path.
    """
    if not records:
        return
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO vector_index_state (chunk_id, file_id, embed_model, content_hash, indexed_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(chunk_id) DO UPDATE SET
                embed_model = excluded.embed_model,
                content_hash = excluded.content_hash,
                indexed_at = CURRENT_TIMESTAMP;
            """,
            records,
        )


def delete_file_vector_state(
    file_id: int,
    embed_model: str | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Delete vector index records for a given file."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        if embed_model is not None:
            cursor.execute(
                "DELETE FROM vector_index_state WHERE file_id = ? AND embed_model = ?;",
                (file_id, embed_model),
            )
        else:
            cursor.execute(
                "DELETE FROM vector_index_state WHERE file_id = ?;",
                (file_id,),
            )


def update_file_indexing_status(
    file_id: int,
    embed_model: str = "BAAI/bge-small-en-v1.5",
    db_path: Path | str | None = None,
) -> None:
    """Update indexed_chunk_count and is_indexed flag on files table."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) as total_chunks,
                   (SELECT COUNT(*) FROM vector_index_state v
                    WHERE v.file_id = ? AND v.embed_model = ?) as indexed_count
            FROM chunks WHERE file_id = ?;
            """,
            (file_id, embed_model, file_id),
        )
        row = cursor.fetchone()
        if row:
            total_chunks = int(row["total_chunks"])
            indexed_count = int(row["indexed_count"])
            is_indexed = 1 if (total_chunks > 0 and indexed_count >= total_chunks) else 0
            cursor.execute(
                """
                UPDATE files
                SET indexed_chunk_count = ?,
                    is_indexed = ?
                WHERE id = ?;
                """,
                (indexed_count, is_indexed, file_id),
            )


def get_vector_index_counts(
    group_id: int | None = None,
    embed_model: str = "BAAI/bge-small-en-v1.5",
    db_path: Path | str | None = None,
) -> dict[str, int]:
    """Return aggregated vector indexing counts for a group or across all groups."""
    counts = {
        "eligible_chunks": 0,
        "indexed_chunks": 0,
        "pending_chunks": 0,
        "indexed_files": 0,
        "total_files": 0,
    }
    with db_session(db_path) as conn:
        cursor = conn.cursor()

        # Eligible chunks (chunks of processed files)
        q_elig = """
            SELECT COUNT(c.id) as cnt
            FROM chunks c
            JOIN files f ON c.file_id = f.id
            WHERE f.status = 'processed'
        """
        params_elig: list[Any] = []
        if group_id is not None:
            q_elig += " AND f.group_id = ?"
            params_elig.append(group_id)
        cursor.execute(q_elig, tuple(params_elig))
        counts["eligible_chunks"] = int(cursor.fetchone()["cnt"])

        # Indexed chunks for this model
        q_idx = """
            SELECT COUNT(v.chunk_id) as cnt
            FROM vector_index_state v
            JOIN files f ON v.file_id = f.id
            WHERE v.embed_model = ?
        """
        params_idx: list[Any] = [embed_model]
        if group_id is not None:
            q_idx += " AND f.group_id = ?"
            params_idx.append(group_id)
        cursor.execute(q_idx, tuple(params_idx))
        counts["indexed_chunks"] = int(cursor.fetchone()["cnt"])

        # Pending chunks
        counts["pending_chunks"] = max(0, counts["eligible_chunks"] - counts["indexed_chunks"])

        # Files
        q_files = "SELECT COUNT(*) as total, SUM(CASE WHEN is_indexed = 1 THEN 1 ELSE 0 END) as indexed FROM files WHERE status = 'processed'"
        params_f: list[Any] = []
        if group_id is not None:
            q_files += " AND group_id = ?"
            params_f.append(group_id)
        cursor.execute(q_files, tuple(params_f))
        frow = cursor.fetchone()
        if frow:
            counts["total_files"] = int(frow["total"] or 0)
            counts["indexed_files"] = int(frow["indexed"] or 0)

    return counts


def get_chunk_with_metadata(
    chunk_id: str,
    db_path: Path | str | None = None,
) -> sqlite3.Row | None:
    """Retrieve full chunk content, location, filename, group_id, and sender."""
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT c.id, c.file_id, c.page, c.text, c.location, c.chunk_index,
                   f.filename, f.group_id, f.sender
            FROM chunks c
            JOIN files f ON c.file_id = f.id
            WHERE c.id = ?;
            """,
            (chunk_id,),
        )
        return cursor.fetchone()

