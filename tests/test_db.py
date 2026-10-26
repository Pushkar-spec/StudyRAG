"""Tests for StudyRAG SQLite database foundation and schema management."""

from pathlib import Path
import sqlite3
import pytest

from core.db import (
    DatabaseError,
    check_fts5_support,
    db_session,
    get_connection,
    get_existing_tables,
    init_db,
)


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Provide a path to a temporary database file."""
    return tmp_path / "test_studyrag.db"


def test_init_db_creates_all_tables_and_indexes(test_db_path: Path):
    """Verify that init_db creates all 8 expected tables/virtual tables and indexes."""
    db_file = init_db(test_db_path)
    assert db_file.exists()

    tables = get_existing_tables(test_db_path)
    expected_tables = {
        "groups",
        "files",
        "chunks",
        "senders",
        "ocr_cache",
        "llm_cache",
        "usage",
        "chunks_fts",
    }
    assert expected_tables.issubset(set(tables)), f"Missing tables: {expected_tables - set(tables)}"

    with get_connection(test_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type = 'index';")
        indexes = {row["name"] for row in cursor.fetchall()}

        expected_indexes = {
            "idx_files_group_status",
            "idx_files_sha256",
            "idx_chunks_file_id",
            "idx_chunks_file_page",
            "idx_ocr_cache_sha256",
            "idx_llm_cache_kind",
        }
        assert expected_indexes.issubset(indexes), f"Missing indexes: {expected_indexes - indexes}"


def test_init_db_is_idempotent_and_preserves_data(test_db_path: Path):
    """Calling init_db multiple times must not raise errors or destroy existing data."""
    init_db(test_db_path)

    # Insert test record
    with db_session(test_db_path) as conn:
        conn.execute("INSERT INTO groups (name) VALUES ('Math Study Group');")

    # Run init_db a second time
    init_db(test_db_path)

    # Verify data is still intact
    with get_connection(test_db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM groups WHERE name = 'Math Study Group';")
        row = cur.fetchone()
        assert row is not None
        assert row["name"] == "Math Study Group"


def test_foreign_key_enforcement_active(test_db_path: Path):
    """Connections must enforce foreign keys (PRAGMA foreign_keys = ON)."""
    init_db(test_db_path)

    with get_connection(test_db_path) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys;")
        assert cur.fetchone()[0] == 1

        # Attempt inserting a file referencing non-existent group_id 999
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO files (group_id, sha256, filename, path, ext)
                VALUES (999, 'abc123hash', 'notes.pdf', '/path/notes.pdf', '.pdf');
                """
            )


def test_unique_group_name_constraint(test_db_path: Path):
    """Groups must have unique names."""
    init_db(test_db_path)

    with db_session(test_db_path) as conn:
        conn.execute("INSERT INTO groups (name) VALUES ('Physics 101');")

    with pytest.raises(sqlite3.IntegrityError):
        with db_session(test_db_path) as conn:
            conn.execute("INSERT INTO groups (name) VALUES ('Physics 101');")


def test_unique_file_per_group_constraint(test_db_path: Path):
    """A file with the same sha256 cannot be added twice to the same group."""
    init_db(test_db_path)

    with db_session(test_db_path) as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO groups (name) VALUES ('Chemistry Group');")
        group_id = cur.lastrowid

        conn.execute(
            """
            INSERT INTO files (group_id, sha256, filename, path, ext)
            VALUES (?, 'hash_chem_001', 'organic.pdf', '/chem/organic.pdf', '.pdf');
            """,
            (group_id,),
        )

    # Attempt inserting same sha256 in same group
    with pytest.raises(sqlite3.IntegrityError):
        with db_session(test_db_path) as conn:
            conn.execute(
                """
                INSERT INTO files (group_id, sha256, filename, path, ext)
                VALUES (?, 'hash_chem_001', 'organic_v2.pdf', '/chem/organic_v2.pdf', '.pdf');
                """,
                (group_id,),
            )


def test_foreign_key_cascade_deletion(test_db_path: Path):
    """Deleting a group should cascade delete its files and chunks."""
    init_db(test_db_path)

    with db_session(test_db_path) as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO groups (name) VALUES ('Biology');")
        group_id = cur.lastrowid

        cur.execute(
            """
            INSERT INTO files (group_id, sha256, filename, path, ext)
            VALUES (?, 'bio_hash', 'cells.pdf', '/bio/cells.pdf', '.pdf');
            """,
            (group_id,),
        )
        file_id = cur.lastrowid

        cur.execute(
            """
            INSERT INTO chunks (id, file_id, page, text)
            VALUES ('chunk_1', ?, 1, 'Mitochondria is the powerhouse of the cell');
            """,
            (file_id,),
        )

    # Delete the group
    with db_session(test_db_path) as conn:
        conn.execute("DELETE FROM groups WHERE id = ?;", (group_id,))

    # Verify cascading deletion
    with get_connection(test_db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM files WHERE id = ?;", (file_id,))
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT COUNT(*) FROM chunks WHERE id = 'chunk_1';")
        assert cur.fetchone()[0] == 0


def test_fts5_virtual_table_creation_and_search(test_db_path: Path):
    """FTS5 table must support full-text MATCH queries."""
    init_db(test_db_path)

    with db_session(test_db_path) as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO groups (name) VALUES ('FTS Group');")
        group_id = cur.lastrowid
        cur.execute(
            """
            INSERT INTO files (group_id, sha256, filename, path, ext)
            VALUES (?, 'fts_hash', 'lecture.pdf', '/fts/lecture.pdf', '.pdf');
            """,
            (group_id,),
        )
        file_id = cur.lastrowid

        # Insert a chunk - the trigger automatically inserts into chunks_fts
        cur.execute(
            """
            INSERT INTO chunks (id, file_id, page, text)
            VALUES ('chunk_fts_1', ?, 1, 'Quantum entanglement is a physical phenomenon observed at quantum scale');
            """,
            (file_id,),
        )

    with get_connection(test_db_path) as conn:
        cur = conn.cursor()
        # Query FTS5 table
        cur.execute("SELECT id FROM chunks_fts WHERE chunks_fts MATCH 'entanglement';")
        results = cur.fetchall()
        assert len(results) == 1
        assert results[0]["id"] == "chunk_fts_1"

        # Match query with non-matching term returns empty
        cur.execute("SELECT id FROM chunks_fts WHERE chunks_fts MATCH 'photosynthesis';")
        assert len(cur.fetchall()) == 0


def test_fts5_triggers_synchronization(test_db_path: Path):
    """Verify triggers synchronize chunks INSERT, UPDATE, and DELETE into chunks_fts."""
    init_db(test_db_path)

    with db_session(test_db_path) as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO groups (name) VALUES ('Trigger Group');")
        group_id = cur.lastrowid
        cur.execute(
            """
            INSERT INTO files (group_id, sha256, filename, path, ext)
            VALUES (?, 'trig_hash', 'doc.pdf', '/doc.pdf', '.pdf');
            """,
            (group_id,),
        )
        file_id = cur.lastrowid

        # 1. Insert chunk
        cur.execute(
            """
            INSERT INTO chunks (id, file_id, page, text)
            VALUES ('chunk_t1', ?, 1, 'Initial text about thermodynamics');
            """,
            (file_id,),
        )

    with get_connection(test_db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM chunks_fts WHERE chunks_fts MATCH 'thermodynamics';")
        assert len(cur.fetchall()) == 1

    # 2. Update chunk
    with db_session(test_db_path) as conn:
        conn.execute(
            """
            UPDATE chunks SET text = 'Revised text about electromagnetism' WHERE id = 'chunk_t1';
            """
        )

    with get_connection(test_db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM chunks_fts WHERE chunks_fts MATCH 'thermodynamics';")
        assert len(cur.fetchall()) == 0
        cur.execute("SELECT id FROM chunks_fts WHERE chunks_fts MATCH 'electromagnetism';")
        assert len(cur.fetchall()) == 1

    # 3. Delete chunk
    with db_session(test_db_path) as conn:
        conn.execute("DELETE FROM chunks WHERE id = 'chunk_t1';")

    with get_connection(test_db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM chunks_fts WHERE chunks_fts MATCH 'electromagnetism';")
        assert len(cur.fetchall()) == 0


def test_ocr_cache_and_confidence_column(test_db_path: Path):
    """Verify ocr_cache table stores text and confidence properly."""
    init_db(test_db_path)

    with db_session(test_db_path) as conn:
        conn.execute(
            """
            INSERT INTO ocr_cache (sha256, page, text, confidence)
            VALUES ('img_hash_1', 1, 'Extracted text via OCR', 0.94);
            """
        )

    with get_connection(test_db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM ocr_cache WHERE sha256 = 'img_hash_1' AND page = 1;")
        row = cur.fetchone()
        assert row is not None
        assert row["text"] == "Extracted text via OCR"
        assert row["confidence"] == 0.94


def test_transaction_rollback_on_error(test_db_path: Path):
    """db_session rolls back uncommitted changes when an exception is raised."""
    init_db(test_db_path)

    with pytest.raises(RuntimeError):
        with db_session(test_db_path) as conn:
            conn.execute("INSERT INTO groups (name) VALUES ('Rollback Group');")
            raise RuntimeError("Forced simulation error")

    with get_connection(test_db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM groups WHERE name = 'Rollback Group';")
        assert cur.fetchone()[0] == 0


def test_check_fts5_support():
    """Verify check_fts5_support function returns True on this environment."""
    assert check_fts5_support() is True


def test_get_file_chunks_ordering_and_retrieval(test_db_path: Path):
    """Verify get_file_chunks retrieves chunks in index order with expected schema fields."""
    from core.db import get_or_create_group, insert_file, get_file_chunks
    init_db(test_db_path)

    gid = get_or_create_group("CS_Chunk_Test", db_path=test_db_path)
    fid = insert_file(gid, "h_chunk", "notes.pdf", "/p/notes.pdf", ".pdf", db_path=test_db_path)

    with get_connection(test_db_path) as conn:
        conn.execute(
            "INSERT INTO chunks (id, file_id, page, text, location, chunk_index) VALUES (?, ?, ?, ?, ?, ?);",
            (f"chk_{fid}_0001", fid, 2, "Second chunk text", "Page 2", 1),
        )
        conn.execute(
            "INSERT INTO chunks (id, file_id, page, text, location, chunk_index) VALUES (?, ?, ?, ?, ?, ?);",
            (f"chk_{fid}_0000", fid, 1, "First chunk text", "Page 1", 0),
        )
        conn.commit()

    chunks = get_file_chunks(fid, db_path=test_db_path)
    assert len(chunks) == 2
    assert chunks[0]["id"] == f"chk_{fid}_0000"
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["text"] == "First chunk text"
    assert chunks[0]["location"] == "Page 1"
    assert chunks[0]["page"] == 1

    assert chunks[1]["id"] == f"chk_{fid}_0001"
    assert chunks[1]["chunk_index"] == 1
    assert chunks[1]["text"] == "Second chunk text"

    # Also supports string file_id
    str_chunks = get_file_chunks(str(fid), db_path=test_db_path)
    assert len(str_chunks) == 2


# ---------------------------------------------------------------------------
# Migration Regression Tests (Legacy Schemas, Backfill, Idempotency)
# ---------------------------------------------------------------------------

def test_migration_fresh_database(tmp_path: Path):
    """A fresh database initializes with all tables, additive columns, and indexes."""
    db_file = tmp_path / "fresh.db"
    init_db(db_file)

    with get_connection(db_file) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(chunks);")
        chunk_cols = {row["name"] for row in cur.fetchall()}
        assert {"id", "file_id", "page", "text", "location", "chunk_index"}.issubset(chunk_cols)

        cur.execute("PRAGMA table_info(files);")
        file_cols = {row["name"] for row in cur.fetchall()}
        assert {"error_message", "chunk_count", "processed_at", "indexed_chunk_count", "is_indexed"}.issubset(file_cols)

        cur.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'idx_chunks_file_chunk_index';")
        assert cur.fetchone() is not None


def test_migration_legacy_database_without_chunk_index(tmp_path: Path):
    """Database created before chunk_index exists safely migrates and backfills ordering."""
    legacy_db = tmp_path / "legacy_step1.db"
    conn = sqlite3.connect(str(legacy_db))
    cur = conn.cursor()

    # Create schema as it existed in Step 1.1 / 1.2
    cur.execute("CREATE TABLE groups (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL);")
    cur.execute("""
    CREATE TABLE files (
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
        UNIQUE(group_id, sha256)
    );
    """)
    cur.execute("""
    CREATE TABLE chunks (
        id TEXT PRIMARY KEY,
        file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        page INTEGER,
        text TEXT NOT NULL
    );
    """)
    cur.execute("INSERT INTO groups (name) VALUES ('LegacyGroup');")
    cur.execute("INSERT INTO files (group_id, sha256, filename, path, ext, status) VALUES (1, 'hash_leg', 'legacy.pdf', '/path.pdf', '.pdf', 'processed');")
    # Insert legacy chunks without location or chunk_index
    cur.execute("INSERT INTO chunks (id, file_id, page, text) VALUES ('chk_1_0000', 1, 1, 'Legacy Page 1 content');")
    cur.execute("INSERT INTO chunks (id, file_id, page, text) VALUES ('chk_1_0001', 1, 2, 'Legacy Page 2 content');")
    cur.execute("INSERT INTO chunks (id, file_id, page, text) VALUES ('chk_1_0002', 1, NULL, 'Legacy Unpaginated appendix');")
    conn.commit()
    conn.close()

    # Run init_db - must NOT fail with "no such column: chunk_index"
    init_db(legacy_db)

    # Verify migration and deterministic backfill
    from core.db import get_file_chunks
    chunks = get_file_chunks(1, db_path=legacy_db)
    assert len(chunks) == 3

    # Preserved text and received deterministic indices
    assert chunks[0]["id"] == "chk_1_0000"
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["location"] == "Page 1"
    assert chunks[0]["text"] == "Legacy Page 1 content"

    assert chunks[1]["id"] == "chk_1_0001"
    assert chunks[1]["chunk_index"] == 1
    assert chunks[1]["location"] == "Page 2"

    assert chunks[2]["id"] == "chk_1_0002"
    assert chunks[2]["chunk_index"] == 2
    assert chunks[2]["text"] == "Legacy Unpaginated appendix"

    # Index exists
    with get_connection(legacy_db) as c:
        cur = c.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'idx_chunks_file_chunk_index';")
        assert cur.fetchone() is not None


def test_migration_legacy_db_with_existing_chunks_and_fts5(tmp_path: Path):
    """FTS5 triggers and search continue working after migration on legacy database."""
    db_file = tmp_path / "legacy_fts.db"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    cur.execute("CREATE TABLE groups (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL);")
    cur.execute("CREATE TABLE files (id INTEGER PRIMARY KEY AUTOINCREMENT, group_id INTEGER, sha256 TEXT, filename TEXT, path TEXT, ext TEXT, status TEXT);")
    cur.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, file_id INTEGER, page INTEGER, text TEXT NOT NULL);")
    cur.execute("INSERT INTO groups (name) VALUES ('FTS_Group');")
    cur.execute("INSERT INTO files (id, group_id, sha256, filename, path, ext, status) VALUES (1, 1, 'h', 'dbms.pdf', '/p', '.pdf', 'processed');")
    cur.execute("INSERT INTO chunks (id, file_id, page, text) VALUES ('chk_1_0000', 1, 1, 'Database relational normalization reduces redundancy.');")
    conn.commit()
    conn.close()

    init_db(db_file)

    from core.db import search_chunks_fts
    results = search_chunks_fts("normalization", db_path=db_file)
    assert len(results) == 1
    assert results[0]["id"] == "chk_1_0000"
    assert "normalization" in results[0]["text"]


def test_migration_db_missing_vector_columns_and_tables(tmp_path: Path):
    """Database missing vector columns on files or vector_index_state table migrates cleanly."""
    db_file = tmp_path / "step2_1_db.db"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    # Step 2.1 state: has chunks(location, chunk_index) but lacks files(indexed_chunk_count, is_indexed) and vector_index_state
    cur.execute("CREATE TABLE groups (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL);")
    cur.execute("""
    CREATE TABLE files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
        sha256 TEXT NOT NULL,
        filename TEXT NOT NULL,
        path TEXT NOT NULL,
        ext TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'new',
        error_message TEXT,
        chunk_count INTEGER DEFAULT 0,
        processed_at TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE chunks (
        id TEXT PRIMARY KEY,
        file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        page INTEGER,
        text TEXT NOT NULL,
        location TEXT,
        chunk_index INTEGER
    );
    """)
    cur.execute("INSERT INTO groups (name) VALUES ('G');")
    cur.execute("INSERT INTO files (id, group_id, sha256, filename, path, ext, status, chunk_count) VALUES (1, 1, 'h', 'doc.pdf', '/p', '.pdf', 'processed', 1);")
    cur.execute("INSERT INTO chunks (id, file_id, page, text, location, chunk_index) VALUES ('chk_1_0000', 1, 1, 'Text', 'Page 1', 0);")
    conn.commit()
    conn.close()

    # Run init_db
    init_db(db_file)

    with get_connection(db_file) as c:
        cur = c.cursor()
        cur.execute("PRAGMA table_info(files);")
        f_cols = {row["name"] for row in cur.fetchall()}
        assert "indexed_chunk_count" in f_cols
        assert "is_indexed" in f_cols

        cur.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'vector_index_state';")
        assert cur.fetchone() is not None


def test_migration_idempotent_repeated_runs(tmp_path: Path):
    """Repeated calls to init_db on migrated databases preserve all records and raise no errors."""
    db_file = tmp_path / "idempotent.db"
    init_db(db_file)
    init_db(db_file)
    init_db(db_file)

    from core.db import get_or_create_group, insert_file, get_existing_tables
    gid = get_or_create_group("RepeatGroup", db_path=db_file)
    fid = insert_file(gid, "h_rep", "test.pdf", "/p", ".pdf", db_path=db_file)

    # Calling again after inserting data
    init_db(db_file)
    init_db(db_file)

    tables = get_existing_tables(db_file)
    assert "groups" in tables
    assert "files" in tables
    assert "chunks" in tables
    assert "vector_index_state" in tables


