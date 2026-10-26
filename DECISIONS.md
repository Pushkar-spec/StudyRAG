# StudyRAG — Architecture & Design Decisions

This document records the architectural decisions, trade-offs, schema choices, and technical foundations established in **StudyRAG Step 1.1**.

---

## 1. Local-First Application Storage Outside Repository

### Decision
Store all user data, SQLite database (`studyrag.db`), vector embeddings (`chroma/`), logs (`app.log`), and persistent settings (`settings.json`) in the user's home directory under `~/.studyrag`, rather than inside the repository folder.

### Rationale
- **Privacy and Data Protection:** WhatsApp study group exports may contain confidential student documents, exam materials, and personal messages. Placing them outside the git workspace eliminates the risk of accidentally committing sensitive user data to source control.
- **Portability:** The codebase can be pulled, updated, or branched without clobbering local user databases or configuration.
- **Configurability:** An override environment variable (`STUDYRAG_HOME`) is supported, allowing developers and test fixtures to redirect the application root directory cleanly without modifying production code.

---

## 2. Configuration Management with Pydantic v2 & JSON

### Decision
Use **Pydantic v2** (`BaseModel`, `Field`, `model_validator`) for type validation and constraints, storing persisted settings in human-readable JSON (`settings.json`).

### Key Behaviors
1. **Defensive Corruption Handling:** If `settings.json` is malformed (invalid JSON syntax) or fails schema validation, `core.config.load()` raises a descriptive `ConfigError`. It **never** silently overwrites or destroys a corrupt settings file with default values.
2. **Atomic Writes:** When saving settings via `core.config.save()`, files are written to a temporary sibling file (`settings.json.tmp`) and atomically replaced (`replace()`), guarding against partial file writes during sudden process termination or power loss.
3. **Decoupling of Secrets:** The Gemini API key and external credentials are intentionally excluded from `settings.json`. Credentials belong in environment variables (`.env`), keeping configuration files safe for inspection, sharing, or backup.
4. **Provisional Model Default:** `gemini_model` defaults provisionally to `gemini-2.5-flash-lite`. As documented in the plan, model availability and rate limits must be validated against the active Gemini API SDK during the subsequent LLM integration step.

---

## 3. SQLite Database Schema & Constraint Architecture

### Decision
Use Python's standard-library `sqlite3` without heavyweight ORMs (e.g. SQLAlchemy), adopting a clean, documented helper API (`init_db`, `get_connection`, `db_session`).

### Schema Structure & Rationales

| Table | Primary Key | Key Constraints & Indexes | Purpose |
|---|---|---|---|
| `groups` | `id INTEGER PK AUTOINCREMENT` | `name TEXT UNIQUE NOT NULL` | Tracks distinct WhatsApp study groups |
| `files` | `id INTEGER PK AUTOINCREMENT` | `FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE`<br>`UNIQUE(group_id, sha256)`<br>Index: `(group_id, status)`, `(sha256)` | Stores metadata, hash, and status of study files |
| `chunks` | `id TEXT PK` | `FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE`<br>Index: `(file_id)`, `(file_id, page)` | Extracted text chunks from documents |
| `chunks_fts` | Virtual Table | `fts5(id UNINDEXED, text)` | Full-text lexical search virtual table |
| `senders` | `(group_id, name)` | `FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE` | Sender whitelist & professor status |
| `ocr_cache` | `(sha256, page)` | Index: `(sha256)` | Cached page-level OCR text and extraction confidence |
| `llm_cache` | `key TEXT PK` | Index: `(kind)` | Deduplicates LLM grading & query results |
| `usage` | `day TEXT PK` | Default 0 counters | Enforces daily token cap and tracks API calls |

### Mandatory Foreign-Key Enforcement
In SQLite, foreign-key constraint enforcement is disabled by default per connection. `core.db.get_connection()` explicitly executes:
```sql
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;
```
This guarantees that cascading deletes (e.g. deleting a group deletes all related files and chunks) and relational integrity are enforced at the engine level for every query.

### Early Inclusion of `ocr_cache.confidence`
The `confidence REAL` column is included in the `ocr_cache` schema from the start, avoiding messy schema migrations in Step 1.4 when the Tesseract OCR pipeline is introduced.

---

## 4. SQLite FTS5 Full-Text Search and Synchronization

### Decision
StudyRAG relies on SQLite's native **FTS5** virtual table module for BM25-based lexical keyword matching.

### Synchronization Approach: Native SQLite Triggers
A core requirement is ensuring `chunks_fts` stays perfectly consistent with `chunks`. Two approaches were evaluated:

1. **Application-Level Synchronization (Rejected for Step 1.1):**
   - Requiring every Python ingestion script to manually insert/delete from both `chunks` and `chunks_fts` within the same transaction.
   - *Drawback:* Prone to inconsistencies if any code path performs bulk operations, updates, or deletes without remembering the FTS index.

2. **Native SQLite Triggers (Selected):**
   - We implement three native triggers directly in the SQLite schema:
     - `chunks_ai AFTER INSERT ON chunks`: Inserts new `(id, text)` into `chunks_fts`.
     - `chunks_ad AFTER DELETE ON chunks`: Deletes corresponding `id` from `chunks_fts`.
     - `chunks_au AFTER UPDATE ON chunks`: Deletes old `id` and inserts updated `(id, text)`.
   - *Advantages:* 
     - **Atomic Consistency:** The database engine maintains the FTS index inside the exact same internal transaction as the primary table operation.
     - **Simplicity:** Code interacting with document chunks only needs to target the `chunks` table; indexing is completely transparent.
     - **Zero Orphan Index Records:** Deleting or replacing chunks immediately updates the virtual index.

### Verification of FTS5 Support
Because some custom or minimal Python distributions omit SQLite FTS5, `check_fts5_support()` executes a temporary virtual table check during `init_db()`. If unavailable, a clear `FTS5NotSupportedError` is raised with actionable instructions rather than failing cryptically later during ingestion.

---

## 5. Idempotent Schema Migration

### Decision
`init_db()` is designed to be fully idempotent:
- All DDL statements use `IF NOT EXISTS` for tables, indexes, and triggers.
- Re-running `init_db()` at application boot or across multiple tests never drops existing tables or resets user data.

---

## 6. WhatsApp Chat Parsing & Format Normalization (Step 1.2)

### Decision
Implement dedicated regex-driven parsing in `core/chat_parser.py` capable of handling both Android (`date, time - Sender: Msg`) and iOS (`[date, time] Sender: Msg`) export formats, multiline message continuations, invisible Unicode marks, and ambiguous date resolution.

### Key Details
1. **Unicode Sanitization:** WhatsApp exports routinely include invisible directionality characters (such as `\u200e` LTR mark, `\u200f` RTL mark, and `\ufeff` BOM), along with narrow no-break spaces (`\u202f`) preceding AM/PM tokens. The parser cleans these markers prior to tokenization to guarantee clean regex matching.
2. **Date Format Detection (DMY vs MDY):** By inspecting all date strings in the chat log, any number $> 12$ immediately resolves the day position unambiguously. When all dates are $\le 12$, the parser falls back to a configurable default (`dmy`, standard in international/UK/India exports) while flagging `is_ambiguous_date = True`.
3. **Attachment Detection:** Recognizes both `<attached: filename>` and `filename (file attached)` syntaxes, extracting the referenced filename while isolating the remaining user message text as the caption.
4. **Chat File Discovery:** When processing exports, `find_chat_file()` prioritizes `.txt` files containing `chat` (case-insensitively), falling back to any `.txt` file while strictly ignoring `.md` files.

---

## 7. Safe Ingestion & ZIP Slip Protection (Step 1.2)

### Decision
Implement `ingest_zip()` and `ingest_folder()` in `core/ingest.py` with strict path traversal defenses, streaming SHA-256 deduplication, and non-destructive file handling.

### Key Details
1. **ZIP Slip Defense:** Archive entry names are validated via `is_safe_zip_entry()`. Absolute paths (leading `/`, `\`, or Windows drive letters) and traversal paths (`..`) that resolve outside the group's target extraction folder are immediately rejected with warnings without writing to disk.
2. **Non-Destructive Processing:** User source archives and original source folders are never modified, moved, or deleted.
3. **Collision Safety:** If a file with the same name already exists in the destination folder:
   - If the SHA-256 hash matches, the existing extracted file is reused.
   - If the SHA-256 hash differs (two different files with the same name), the newly extracted file is safely disambiguated with a numeric suffix (e.g. `doc_1.pdf`) so no data is overwritten.
4. **Supported Document Filtering:** Only `.pdf`, `.pptx`, `.docx`, `.jpg`, `.jpeg`, and `.png` files are registered. Image files are preserved on disk and recorded for the future OCR processing phase.
5. **Zero-Document Export Warning:** If an export contains zero supported documents (common when users export WhatsApp chats without media), a clear warning is returned directing the user to export with media or use folder import.

---

## 8. Document-to-Message Linking & Context Window (Step 1.2)

### Decision
Link extracted study documents to WhatsApp chat messages and surrounding context to preserve teacher notes and exam instructions.

### Key Details
1. **Surrounding Context Window:** For each attachment message, surrounding messages from the **same sender** within a **5-minute window** ($\le 300$ seconds) are collected, capped at a maximum of **3 surrounding messages**. Unrelated messages from other chat participants within the window are strictly excluded.
2. **Ambiguity Defense:** If multiple attachment messages or multiple files share the exact same filename, automated caption assignment is skipped with an explanatory warning to prevent assigning the wrong teacher's notes or instructions to a mismatched file.
3. **Preservation of Senders & Roles:** Senders are recorded in the `senders` table with `ON CONFLICT (group_id, name) DO NOTHING`. Reimporting an export preserves existing `is_professor` flags.

---

## 9. Persistent Attachment Reference Tracking & Recovery (Step 1.3)

### Decision
Store all WhatsApp chat attachment references in a dedicated `attachment_refs` table in SQLite, enabling lifecycle tracking, status management, later recovery of missing documents, and manual disambiguation.

### Schema Design & Deduplication
- **Identity & Unique Key:** Each attachment reference is uniquely identified by `(group_id, dedup_key)`. The `dedup_key` is a deterministic SHA-256 hash computed from `group_id`, `sender`, `sent_at`, normalized filename, and the initial 150 characters of message text.
- **Idempotency:** Reimporting the same chat log updates existing records rather than producing duplicate rows, preserving manual resolutions.
- **Traceability:** Tracks `original_filename`, `normalized_filename`, `sender`, `sent_at`, `caption`, `context`, `file_id`, `status`, `resolution_method`, and `resolution_note`.

---

## 10. Attachment Status Transitions & Lifecycle

### Status Definitions
1. **`missing`**: The chat log explicitly references an attachment (e.g. `<attached: DBMS_Unit_5.pdf>`) or indicates omitted media, but no matching file exists in the export archive or group storage.
2. **`available`**: An unambiguous matching file exists in group storage and has been successfully registered in the `files` table (`file_id` populated).
3. **`ambiguous`**: Multiple candidate files on disk or multiple chat messages referencing the same filename prevent automated linking without risk of misattribution.
4. **`unsupported`**: An attachment exists in the export, but its file type (e.g. `.zip`, `.exe`, `.mp4`) is not currently supported for study document indexing.
5. **`import_failed`**: The attachment was present in the archive, but extraction or database insertion failed due to I/O or archive errors.

### Transition Rules
- `missing -> available`: Occurs automatically when the student downloads the missing document later and imports it via `recover_file_or_folder()`, provided the match is unambiguous.
- `missing -> ambiguous`: Occurs if multiple candidate files with the same normalized name are imported.
- `ambiguous -> available`: Occurs when the user reviews candidate files in the Streamlit UI and clicks "Link Selected File to Reference", recording `resolution_method = 'manual'`.
- `available` preservation: A reference marked `manual` is never overridden by subsequent automatic re-imports.

---

## 11. Matching Confidence & Group Isolation

### Rules
1. **Matching Priority Order:**
   - 1. Exact filename match within the target group.
   - 2. Unicode NFKC-normalized, case-insensitive filename match within the target group.
   - 3. Manual user disambiguation.
2. **Group Isolation:** References are strictly scoped by `group_id`. Automatic cross-group matching is prohibited to prevent leaking documents across courses. Manual linking validates that `ref.group_id == file.group_id`.
3. **Registration vs Indexing:** An `available` attachment status confirms only that the file is present and registered in SQLite. Document text extraction, chunking, embeddings, and vector indexing remain deferred to subsequent steps.

---

## 12. Chunk Size and Overlap Defaults (Step 2.1)

### Decision
Set default document chunk size to `1000` characters and overlap to `200` characters, exposed as validated Pydantic settings in `core.config.Settings`:
- `chunk_size: int = Field(default=1000, ge=1)`
- `chunk_overlap: int = Field(default=200, ge=0)`
- Interdependent validation enforces: `chunk_size > 0`, `chunk_overlap >= 0`, and `chunk_overlap < chunk_size`.

### Rationale
- `1000` characters (approximately 150-250 words) comfortably accommodates standard academic paragraphs and PowerPoint slide notes without truncation.
- `200` characters overlap (2-3 sentences) prevents contextual fragmentation across chunk boundaries, ensuring keywords and definitions that span boundaries can still be matched during retrieval.
- Boundary-aware splitting prioritizes paragraph breaks (`\n\n`), sentence boundaries (`. `, `? `, `! `), and whitespace, avoiding mid-word cutting and never emitting empty chunks.

---

## 13. Source-Location Conventions & Backward Compatibility

### Decision
Preserve physical page numbers for paginated formats while providing human-readable location schemes for non-paginated documents:
- **PDF:** `page = <int 1-indexed page>`, `location = "Page <N>"`.
- **PowerPoint:** `page = <int 1-indexed slide>`, `location = "Slide <N>"`.
- **Word:** `page = <int block_index>`, `location = "Paragraph <N>"` or `"Table <N>"`.

### Database Schema Compatibility
- The existing SQLite `chunks` table had `page INTEGER`.
- Additive columns `location TEXT` and `chunk_index INTEGER` were introduced using safe idempotent migrations (`PRAGMA table_info`).
- `chunk_index` (0, 1, 2...) guarantees deterministic chunk ordering within each document.
- `id` follows a deterministic pattern: `f"chk_{file_id}_{chunk_index:04d}"`.

---

## 14. File Extraction Processing Statuses

### Statuses
1. **`new`**: Registered document pending text extraction.
2. **`processed`**: Successfully extracted and chunked. Chunks are active in `chunks` and `chunks_fts`.
3. **`empty`**: Successful extraction attempt, but no selectable text was found (e.g. scanned image-only PDF, blank slides). Flagged with diagnostic message explaining that OCR is required.
4. **`unsupported`**: Registered document with an extension not supported for text extraction (e.g. `.png`, `.jpg`, `.mp4`).
5. **`failed`**: Extraction or persistence failed (e.g. corrupt archive, password-protected PDF, disk read error). Error message stored in `files.error_message`.

---

## 15. Incremental Processing, Idempotency, and Fault Isolation

### Decision
- **Incremental Processing:** By default, batch processing queries only files with `status IN ('new', 'failed')`. Unchanged `processed` files are skipped without redundant computation or chunk duplication.
- **Transactional Replacement:** Reprocessing a file executes `DELETE FROM chunks WHERE file_id = ?` followed by `INSERT INTO chunks` inside an atomic transaction. If an error occurs midway, the transaction rolls back, preserving previously valid chunks.
- **Fault-Tolerant Batching:** If a corrupt or unsupported file is encountered during batch processing, the error is isolated to that specific file record (`status = 'failed'`). The batch continues processing remaining documents without crashing the pipeline.

---

## 16. SQLite FTS5 Trigger Synchronization & Query Sanitization

### Decision
- **Trigger-Driven Sync:** Triggers `chunks_ai`, `chunks_ad`, and `chunks_au` maintain the virtual `chunks_fts` table transparently whenever rows in `chunks` are inserted, deleted, or updated. No manual duplicate indexing is performed in Python.
- **Query Sanitization:** SQLite FTS5 interprets characters like hyphens (`-`), asterisks (`*`), colons (`:`), and parentheses as operators or column queries. `core.db.sanitize_fts5_query()` sanitizes user queries containing special characters into exact phrases (e.g. `"Entity-Relationship"`), preventing syntax errors.

---

## 17. Local Embedding Model & Singleton Management (Step 2.2)

### Decision
Use **`BAAI/bge-small-en-v1.5`** via the `sentence-transformers` library, loaded and cached in a lazy singleton manager (`core.vector_store.LocalEmbeddingManager`).

### Rationale
- **Performance & Accuracy:** BGE-small-en-v1.5 outputs high-quality 384-dimensional dense vectors with strong academic retrieval performance, low memory footprint (~130MB), and fast CPU inference on standard student laptops.
- **Lazy Initialization:** The model is not loaded at application startup or when only running keyword search. It initializes only when embedding generation or semantic search is explicitly invoked.
- **Test Isolation via Custom Embedder:** `LocalEmbeddingManager.embed_texts` and `embed_query` support an optional `custom_embedder` callable. This enables fast, deterministic unit and integration testing without downloading model weights or making network calls during CI or local testing.
- **Local Privacy Guarantee:** All embeddings are computed locally on the student's hardware. No chunk text or embeddings are sent to external APIs.

---

## 18. Persistent ChromaDB & Cosine Distance Space

### Decision
Use **ChromaDB** with `PersistentClient` targeting `~/.studyrag/chroma/`, configuring the collection with `metadata={"hnsw:space": "cosine"}`.

### Distance Semantics
- ChromaDB cosine space computes cosine distance: $\text{distance} = 1.0 - \text{cosine\_similarity}$.
- For normalized vectors, distance values range from $0.0$ (exact match) to $2.0$ (opposite direction).
- Cosine similarity is derived cleanly as: $\text{similarity} = \max(0.0, 1.0 - \text{distance})$.
- StudyRAG never represents raw distance as a percentage or probability of relevance.

---

## 19. Authoritative SQLite Architecture & Vector Index State Tracking

### Decision
Maintain **SQLite as the sole authoritative source of truth** for all document chunks, file metadata, and extraction records. ChromaDB is treated strictly as an auxiliary similarity index.

### Synchronization Pattern
Because SQLite and ChromaDB cannot participate in a shared atomic transaction, synchronization is designed defensively:
1. **Stable Vector IDs:** Vectors in ChromaDB use the exact stable SQLite chunk ID (`chk_{file_id}_{chunk_index:04d}`).
2. **Dedicated State Table:** The additive table `vector_index_state` in SQLite tracks:
   ```sql
   CREATE TABLE IF NOT EXISTS vector_index_state (
       chunk_id TEXT PRIMARY KEY,
       file_id INTEGER NOT NULL,
       embed_model TEXT NOT NULL,
       content_hash TEXT NOT NULL,
       indexed_at TEXT NOT NULL DEFAULT (datetime('now')),
       FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
   );
   ```
3. **Content-Hash Change Detection:** Before embedding, each chunk's SHA-256 hash is compared to `content_hash` in `vector_index_state`. Unchanged chunks are skipped.
4. **Stale Vector Defense:** When semantic search retrieves candidate chunk IDs from ChromaDB, each ID is immediately resolved against SQLite via `get_chunk_with_metadata()`. If a chunk was modified or deleted in SQLite, it is safely omitted from search results.
5. **Partial Failure Recovery:** If a batch upsert fails, the error is captured and pending chunks remain unrecorded in `vector_index_state`, allowing subsequent retry runs to complete indexing.

---

## 20. Hybrid Candidate Retrieval, Origin Tagging & Score Preservation

### Decision
Combine SQLite FTS5 lexical keyword matching and ChromaDB semantic search in `search_chunks_hybrid(query, limit=...)` while preserving each search method's score on its native scale.

### Key Behaviors
1. **No Artificial Score Blending:** BM25/rank scores from FTS5 and cosine distances from ChromaDB represent fundamentally different mathematical distributions. Attempting to add or multiply them directly without training data causes score distortion.
2. **Candidate Origin Tagging:** Every merged candidate is tagged with its discovery source:
   - `both`: Discovered independently by both keyword and semantic search.
   - `keyword`: Discovered exclusively by SQLite FTS5.
   - `semantic`: Discovered exclusively by ChromaDB cosine search.
3. **Consensus Prioritization:** Candidates matching `both` methods represent strong consensus hits and are prioritized ahead of single-source hits in candidate ranking.
4. **Downstream Readiness:** The resulting `HybridSearchResult` objects preserve `keyword_score`, `semantic_distance`, and `semantic_similarity` as independent features ready for Step 2.3 cross-encoder reranking.

---

## 21. Graceful Semantic-to-Lexical Fallback Strategy

### Decision
If ChromaDB is unavailable, corrupted, or Sentence Transformers fails to load, `search_chunks_hybrid()` catches the exception, logs a descriptive warning, and gracefully returns SQLite FTS5 keyword results.

### Rationale

---

## 22. Safe Ordered Database Migrations & Legacy Chunk Backfills

### Decision
Ensure SQLite schema initialization via `init_db()` strictly executes operations in dependency order with guarded additive migrations and deterministic data backfilling:
1. **Base Tables (`SCHEMA_STATEMENTS`):** Core tables are created first using `CREATE TABLE IF NOT EXISTS`.
2. **Guarded Additive Migrations (`ALTER TABLE ... ADD COLUMN`):** Columns added in later versions (e.g. `files.sender`, `files.sent_at`, `files.caption`, `files.error_message`, `files.chunk_count`, `files.processed_at`, `files.indexed_chunk_count`, `files.is_indexed`, `chunks.location`, `chunks.chunk_index`) are inspected dynamically via `PRAGMA table_info()` and added before any dependent indexes or queries.
3. **Deterministic Backfill for Legacy Data:**
   - Any legacy chunks predating Step 2.1 that lack `chunk_index` are deterministically assigned 0-indexed chunk numbers partitioned by `file_id` ordered by physical page and chunk ID.
   - Legacy chunks with `page` populated receive human-readable location metadata (e.g., `"Page <N>"`).
   - Parent file records update their `chunk_count` to reflect existing stored chunks.
4. **Dependent Indexes (`INDEX_STATEMENTS`):** Indexes referencing additive columns (specifically `idx_chunks_file_chunk_index ON chunks(file_id, chunk_index)`) are created only after the required columns are guaranteed to exist.
5. **Virtual Tables & Triggers (`FTS_STATEMENTS`):** FTS5 virtual tables and synchronization triggers are created and synchronized safely with any existing chunk text.

### Rationale
In earlier versions, `INDEX_STATEMENTS` ran prior to additive migrations. When StudyRAG started on an existing database created prior to Step 2.1, SQLite raised `sqlite3.OperationalError: no such column: chunk_index` during index creation. Enforcing strict DDL ordering and deterministic backfill eliminates migration crashes and preserves all existing user document records.
