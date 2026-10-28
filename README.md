# StudyRAG

**StudyRAG** is a local-first Python application designed to intelligently search and surface exam-relevant study materials (lecture notes, assignments, mock tests, and handouts) from exported WhatsApp study groups.

---

## Key Features

- **Local-First Architecture:** Keeps user data, exported chats, databases, and configuration on the local machine outside the repository.
- **Robust Configuration Management:** Powered by Pydantic v2 validation with persistent JSON settings in `~/.studyrag/settings.json`.
- **Relational & Search Foundation:** SQLite database featuring foreign key enforcement, idempotent schema migration, and native **FTS5** virtual table indexing for full-text retrieval.
- **WhatsApp Ingestion & Chat-Linking Pipeline (Step 1.2):**
  - Android and iOS chat export parser with multiline message handling and Unicode sanitization.
  - Attachment detection and 5-minute surrounding context window linkage.
  - Safe ZIP archive ingestion with **ZIP Slip** path traversal prevention.
  - Recursive local folder import (supports `.pdf`, `.pptx`, `.docx`, `.jpg`, `.jpeg`, `.png`).
  - Streaming SHA-256 deduplication per group.
- **Persistent Missing Attachment Tracking & Recovery (Step 1.3):**
  - Persistent tracking of all referenced WhatsApp attachments in SQLite (`attachment_refs`).
  - Explicit lifecycle statuses: `missing`, `available`, `ambiguous`, `unsupported`, `import_failed`.
  - Later-download recovery workflow reconciling newly imported files to existing chat references.
  - Ambiguity defense with interactive manual candidate matching in the Streamlit UI.
  - Comprehensive handling and advisory warnings for text-only/incomplete WhatsApp exports.
- **Document Text Extraction & Source-Aware Chunking (Step 2.1):**
  - Page-by-page PDF extraction using **PyMuPDF** (`fitz`).
  - Slide-by-slide PowerPoint extraction using **python-pptx** with recursive shape/table traversal.
  - Paragraph- and table-ordered Word extraction using **python-docx**.
  - Configurable, deterministic chunking (default 1000 characters, 200 character overlap) preserving paragraph and sentence boundaries.
  - Full-text search (FTS5) automatic indexing with transactional trigger synchronization.
  - Incremental processing tracking (`new`, `processed`, `empty`, `unsupported`, `failed`) and single-file retry/reprocessing.
- **Local Embeddings, ChromaDB & Hybrid Candidate Retrieval (Step 2.2):**
  - Strictly local semantic embeddings using **Sentence Transformers** (`BAAI/bge-small-en-v1.5`) cached via lazy singleton manager.
  - Persistent local vector store via **ChromaDB** with cosine distance space (`~/.studyrag/chroma/`).
  - Authoritative SQLite traceability: full documents and chunk text reside in SQLite; vectors reference stable chunk IDs (`chk_{file_id}_{chunk_index:04d}`).
  - Incremental, idempotent vector synchronization with SHA-256 content-hash invalidation and `vector_index_state` tracking.
  - Unified hybrid candidate retrieval combining SQLite FTS5 lexical matching and ChromaDB semantic search with deduplication, search origin tagging (`keyword`, `semantic`, `both`), and independent score preservation.
  - Graceful fallback: If ChromaDB or embedding model fails, search automatically falls back to SQLite FTS5 keyword retrieval.
- **Local Cross-Encoder Reranking & Document-Level Ranking (Step 3.2):**
  - High-precision cross-encoder reranking using **Sentence Transformers** (`BAAI/bge-reranker-base`).
  - Lazy loading singleton (`LocalRerankerManager`) with local cache in `~/.cache/huggingface/hub/`.
  - Joint query-passage attention scoring with strict preservation of original chunk metadata and retrieval signals.
  - Bounded, non-inflationary document-level score aggregation with near-duplicate suppression.
  - Returns ranked documents with traceable supporting chunks, source locations (pages/slides), and short excerpts.
  - Transparent fallback: Preserves hybrid candidate ranking if the cross-encoder is disabled or fails.
- **Modular Streamlit Interface:** Modern dark-mode UI with foundation diagnostics, ingestion pipeline, attachment recovery review, document extraction controls, interactive vector indexing, and hybrid & ranked document search.

---

## Directory Structure

```text
StudyRAG/
├── app.py                  # Streamlit entry point (ingestion, recovery, extraction & search)
├── requirements.txt        # Dependencies (PyMuPDF, python-pptx, python-docx, chromadb, sentence-transformers)
├── .env.example            # Template for environment variables and secrets
├── .gitignore              # Repository exclusions (caches, logs, DBs, venv)
├── README.md               # Project documentation & setup instructions
├── DECISIONS.md            # Architecture & design decisions record
├── core/
│   ├── __init__.py
│   ├── config.py           # Pydantic v2 settings & path resolution
│   ├── db.py               # SQLite schema, FTS5 virtual tables, vector_index_state & DB operations
│   ├── chat_parser.py      # WhatsApp Android/iOS chat parser & timestamp normalizer
│   ├── ingest.py           # Ingestion pipeline, attachment tracking & recovery workflow
│   ├── extraction.py       # Document text extraction (PDF, PPTX, DOCX) & chunking pipeline
│   ├── vector_store.py     # Local embeddings, ChromaDB collection & hybrid retrieval
│   ├── reranker.py         # Local cross-encoder reranker (BAAI/bge-reranker-base)
│   └── document_ranking.py # Document-level aggregation & end-to-end ranking orchestrator
├── tests/
│   ├── __init__.py
│   ├── test_config.py      # Unit tests for settings validation & persistence
│   ├── test_db.py          # Unit tests for SQLite schema, constraints & FTS5
│   ├── test_chat_parser.py # Unit tests for Android/iOS formats, dates & attachments
│   ├── test_ingest.py      # Unit tests for ZIP Slip, deduplication & message linking
│   ├── test_recovery.py    # Unit tests for missing attachment tracking & recovery
│   ├── test_extraction.py  # Unit tests for PDF, PPTX, DOCX extraction & chunking
│   ├── test_vector_store.py# Unit tests for ChromaDB, incremental embeddings & hybrid search
│   ├── test_reranker.py    # Unit tests for cross-encoder reranking & pair construction
│   ├── test_document_ranking.py # Tests for document aggregation & end-to-end orchestrator
│   └── test_ui.py          # Streamlit UI AppTest and count reconciliation tests
└── .streamlit/
    └── config.toml         # Streamlit server config (2000 MB upload limit)
```

---

## Application Data Storage

All persistent application data and settings are stored **outside** the repository directory in the user's home folder:

| Asset | Default Path | Purpose |
|---|---|---|
| Application Home | `~/.studyrag/` | Root folder for StudyRAG runtime assets |
| Settings File | `~/.studyrag/settings.json` | Persistent JSON settings (validated with Pydantic v2) |
| Database | `~/.studyrag/studyrag.db` | Primary SQLite database with FTS5 virtual table |
| Vector Store | `~/.studyrag/chroma/` | ChromaDB vector database directory (future phases) |
| Application Log | `~/.studyrag/app.log` | Diagnostic and error logs (future phases) |

> **Note:** You can override the application home location by setting the `STUDYRAG_HOME` environment variable (e.g. `STUDYRAG_HOME=D:\StudyRAGData`).

---

## Prerequisites

- **Operating System:** Windows 10/11 (or macOS / Linux)
- **Python:** Python 3.11 or newer (Python 3.12+ recommended)
- **SQLite:** Python build with SQLite FTS5 extension enabled (standard on official Python distributions)

---

## Setup & Installation on Windows

### 1. Clone or Open the Workspace

Open PowerShell in the project directory:

```powershell
cd c:\Users\<username>\Desktop\StudyRAG
```

### 2. Create and Activate Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If script execution is restricted on Windows PowerShell, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> .\.venv\Scripts\Activate.ps1
> ```

### 3. Install Dependencies

Install the baseline dependencies for Step 1.1:

```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables (Optional for Step 1.1)

Copy `.env.example` to `.env`:

```powershell
cp .env.example .env
```

*Note: In Step 1.1, API keys are not required. A placeholder is provided for later integration.*

---

## Missing-Attachment Tracking & Recovery (Step 1.3)

### Why Track Missing Attachments?
WhatsApp phone exports often omit media files unless "Export with Media" is explicitly chosen. Furthermore, users frequently encounter chat messages referencing lecture notes, slides, or assignments where the media was not downloaded locally at the time of export. 

StudyRAG parses attachment mentions directly from the chat log and persists them in the `attachment_refs` database table. This guarantees that:
- You know exactly which documents were referenced in class.
- The teacher's instructions, timestamps, and surrounding messages are captured even before the document is on disk.
- When you download the missing file later, StudyRAG automatically or manually links it to the original conversation.

### Attachment Statuses

| Status | Meaning |
|---|---|
| `missing` | The chat mentions an attachment, but the document was not found in the export or group storage. |
| `available` | An unambiguous matching document is present and successfully registered in SQLite (`file_id` assigned). |
| `ambiguous` | Multiple files or multiple messages share the same filename. Kept unresolved until manual user selection. |
| `unsupported` | The file was found in the archive, but its extension (e.g. `.zip`, `.exe`, `.mp4`) is not supported for document indexing. |
| `import_failed` | The file was found in the archive, but disk extraction or database registration failed. |

### How to Recover a File Downloaded Later
1. Open the StudyRAG dashboard (`streamlit run app.py`).
2. Navigate to the **Missing Attachment Tracking & Recovery** section.
3. Under **Import Later Download**, select your target study group and specify the path to your downloaded document (e.g. `DBMS_Unit_5.pdf`) or containing folder.
4. Click **Recover Document(s)**. StudyRAG will:
   - Copy the file into your local StudyRAG data directory safely.
   - Register the file with SHA-256 deduplication.
   - Automatically reconcile matching unresolved references, updating status to `available` and linking the original teacher caption to the file.

### Manual Resolution
If multiple candidate files exist (marked as `ambiguous`) or if a WhatsApp export omitted the filename:
1. Open the **Manual Match & Reconcile** tab.
2. Select the unresolved reference.
3. Select the intended registered file from the candidate dropdown.
4. Click **Link Selected File to Reference**. The link is persisted as `manual` and will not be overwritten by future automated re-imports.

### WhatsApp Export Limitations
- **Phone Gallery vs Export:** Seeing a document in your phone's file manager or gallery does not guarantee WhatsApp bundled it into the `.zip` archive.
- **Text-Only Exports:** If an export is created without media, StudyRAG extracts all attachment references and marks them as `missing`, preserving context so you can import the files later.
- **Local-First Boundary:** StudyRAG never connects to WhatsApp servers or web interfaces; all downloads are handled by the user.

### Distinction: Registration vs Indexing
Marking a file as `available` confirms that the file is present in local group storage and registered in SQLite. It does **not** mean text extraction, chunking, or vector embedding have been performed yet; those tasks are handled in subsequent indexing steps.

---

## Document Text Extraction and Chunking (Step 2.1)

### Supported Document Formats
- **PDF (`.pdf`):** Extracted page-by-page via **PyMuPDF** (`fitz`).
- **PowerPoint (`.pptx`):** Extracted slide-by-slide via **python-pptx** (including text boxes, tables, and grouped shapes).
- **Word (`.docx`):** Extracted paragraph-by-paragraph and table-by-table via **python-docx**, preserving document reading order.
- **Plain Text (`.txt`):** Directly read and chunked.

### Source Location Preservation
Every chunk retains precise source metadata:
- **PDF:** Retains the 1-indexed `page` and location string `Page <N>`.
- **PowerPoint:** Retains the 1-indexed `page` (slide number) and location string `Slide <N>`.
- **Word:** Retains the sequential block/paragraph index and location string `Paragraph <N>` or `Table <N>`.

### Configurable Chunking
Configured in `~/.studyrag/settings.json` or overridden via Pydantic settings:
- `chunk_size` (default: `1000` chars): Target maximum chunk size.
- `chunk_overlap` (default: `200` chars): Overlap between consecutive chunks.
- Splitting respects paragraph (`\n\n`) and sentence (`. `, `? `, `! `) boundaries, avoiding mid-word cuts and never producing empty chunks.

### Incremental Processing & Idempotency
- **Status Lifecycle:** `new` (pending) -> `processed` (extracted & chunked), `empty` (no extractable text), `unsupported` (format unsupported), `failed` (corrupt/encrypted).
- **Idempotency:** Re-running processing on unchanged files skips work. Forced re-processing or single-file reprocess safely replaces old chunks and syncs FTS5 in a single atomic SQLite transaction.
- **Fault-Tolerant Batching:** If one document in a study group fails (e.g. corrupt archive entry), the error is isolated and recorded, allowing all valid documents to finish processing.

### Known Limitations
- **No OCR in Step 2.1:** Scanned/image-only PDFs (containing images of pages rather than selectable text) are detected and classified as `empty` with diagnostic messages. OCR capabilities will be added in later steps.
- **Image Files:** Image files (`.png`, `.jpg`, `.jpeg`) do not contain selectable text and are marked `unsupported` until OCR integration.

---

## Local Embeddings, ChromaDB & Hybrid Retrieval (Step 2.2)

### Architecture
StudyRAG couples lexical keyword search (SQLite FTS5) with semantic vector search (ChromaDB + Sentence Transformers) to reliably surface academic materials regardless of whether a student's query matches verbatim textbook terms or conceptually equivalent paraphrases:

1. **Local Embedding Engine:**
   - Model: `BAAI/bge-small-en-v1.5` (via `sentence-transformers`).
   - Dimension: 384-dimensional dense vectors, L2 normalized.
   - Execution: Fully local on CPU/GPU via PyTorch. Never makes cloud API or OpenAI/Gemini calls.
   - Singleton Management: Loaded lazily via `LocalEmbeddingManager` so model weights are cached in memory across queries and indexing batches.

2. **Persistent Local Vector Store:**
   - Backend: **ChromaDB** (`chromadb.PersistentClient`) pointing to `~/.studyrag/chroma/`.
   - Distance Metric: Cosine distance (`hnsw:space = "cosine"`).
   - Authoritative Storage: SQLite remains the single source of truth for full text and document records. ChromaDB vectors store minimal metadata (`chunk_id`, `file_id`, `group_id`, `page`, `location`, `content_hash`, `embed_model`) strictly for fast similarity lookup.

3. **Incremental Synchronization & Change Invalidation:**
   - Chunks are assigned stable deterministic IDs: `chk_{file_id}_{chunk_index:04d}`.
   - An index state table in SQLite (`vector_index_state`) tracks `(chunk_id, file_id, embed_model, content_hash, indexed_at)`.
   - Before embedding, each chunk's SHA-256 content hash is compared against the indexed hash. Chunks with identical hashes and embedding models are skipped automatically (idempotent).
   - If chunk text or embedding model changes, the chunk is re-embedded and upserted.
   - If an embedding batch fails, errors are isolated and recorded without halting the entire process, allowing subsequent retry runs to seamlessly recover pending chunks.

4. **Semantic Search (`search_chunks_semantic`):**
   - Encodes the user query using the same embedding model.
   - Queries ChromaDB collection for nearest neighbors within cosine space.
   - Resolves all candidate chunk IDs against SQLite. If a chunk was deleted or modified in SQLite, stale vector entries are automatically skipped.
   - Returns structured `SemanticSearchResult` with normalized cosine distance and cosine similarity (`1.0 - distance`).

5. **Unified Hybrid Candidate Retrieval (`search_chunks_hybrid`):**
   - Executes parallel FTS5 BM25 search and ChromaDB cosine vector search.
   - Deduplicates candidates by stable `chunk_id`.
   - Tags search origin:
     - `both`: Consensus hit matching both keyword search and semantic vector similarity.
     - `keyword`: Discovered via SQLite FTS5 lexical matching.
     - `semantic`: Discovered via ChromaDB semantic vector search.
   - Preserves independent scores: Retains `keyword_score` (inverse BM25 rank score) and `semantic_distance` / `semantic_similarity` separately. Never mathematically distorts or equates raw keyword rank and vector distance.
   - Prioritizes consensus hits (`both`) while keeping diverse candidate results ready for Step 2.3 reranking.

6. **Graceful Degradation / Fallback:**
   - If ChromaDB is uninitialized, corrupted, or Sentence Transformers fails to load, StudyRAG automatically falls back to SQLite FTS5 keyword retrieval with diagnostic warnings in the logs. Keyword search remains fully operational at all times.

---

## Local Cross-Encoder Reranking & Document-Level Ranking (Step 3.2)

### Bi-Encoder Embeddings vs Cross-Encoder Reranking

| Aspect | Bi-Encoder (`BAAI/bge-small-en-v1.5`) | Cross-Encoder (`BAAI/bge-reranker-base`) |
|---|---|---|
| **Role** | Broad candidate retrieval (top 20–50 chunks). | High-precision candidate reranking & scoring. |
| **Architecture** | Encodes query and passages into independent dense vectors. | Encodes query and passage jointly via self-attention: `CrossEncoder([query, text])`. |
| **Speed & Scalability** | Sub-millisecond vector similarity search across thousands of chunks. | Higher computational cost; ideal for reranking top hybrid candidates. |
| **Semantic Nuance** | Captures general topical similarity. | Resolves subtle term dependencies, negation, and direct relevance. |

### Local Model Caching & Lifecycle
- **Offline Inference:** The cross-encoder runs strictly locally via `sentence-transformers.CrossEncoder`. No external API keys or remote inference calls are ever made.
- **Cache Location:** Automatically downloaded from Hugging Face hub on first use and cached in `~/.cache/huggingface/hub/models--BAAI--bge-reranker-base`.
- **Lazy Loading Singleton:** Managed by `LocalRerankerManager`. The model is instantiated only when a search request requires it and remains in memory across subsequent Streamlit queries.
- **Device Support:** Automatically selects `cuda` if compatible; falls back cleanly to `cpu`. Configurable via `rerank_device`.

### Document-Level Bounded Score Aggregation
Candidate chunks are grouped by original `file_id`. The document-level relevance score is computed as:
$$\text{Document Score} = s_1 + \sum_{i=2}^{\min(N, 4)} w_i \cdot \max(0.0, s_i)$$

- **Primary Anchor ($s_1$):** The highest reranker score among chunks from the document.
- **Bounded Bonus ($s_2, s_3, s_4$):** Up to 3 additional distinct chunks contribute with diminishing weights $w = [0.10, 0.05, 0.02]$.
- **Near-Duplicate Suppression:** Chunks with $\ge 60\%$ word overlap (Jaccard similarity) are treated as redundant and award zero bonus, preventing repeated boilerplate from inflating scores.
- **Non-Inflationary Guarantee:** A 100-page document with many marginal chunks cannot outrank a focused 2-page document with a highly relevant chunk.

### Understanding Reranker Scores
- **Relative Relevance Logits:** Cross-encoder scores are raw uncalibrated model logits (e.g. ranging from negative values up to $+10$ or higher).
- **Not Probabilities:** Scores are displayed as **Model Score**, not percentages or confidence probabilities. They indicate relative ranking priority among candidates.

### Graceful Fallback Strategy
If the cross-encoder is disabled (`disable_reranker=True`), encounters an initialization issue, or fails during inference:
1. The hybrid candidate chunks from SQLite FTS5 and ChromaDB are preserved.
2. Documents are ranked deterministically by retrieval signals (`both` consensus hits first, followed by highest semantic similarity or lexical rank).
3. The UI flags the results with `is_reranked = False` and displays transparent diagnostic messages explaining the fallback reason.

---

## WhatsApp Context Retrieval & Filtering (Phase 4, Step 4.1)

### Conversational Provenance for Academic Documents
In collaborative WhatsApp study groups, lecture notes, syllabus drafts, and problem sets are distributed alongside conversational messages from instructors or peers (e.g. *"Here are the updated slides for tomorrow's midterm"*). Step 4.1 extracts and links this conversational provenance to ranked search results.

### Features
1. **Direct Sharing Message & Surrounding Context:**
   - **Sharing Message:** The exact message accompanying the attachment (`caption`).
   - **Surrounding Conversation:** A bounded 5-minute conversational window (up to 3 preceding messages from the sender) capturing context, instructions, or exam warnings.
   - **Provenance Tracking:** Shows original filename, resolution status (`available`, `ambiguous`, `missing`), and resolution method (`direct`, `sha256`, `filename`, `manual`).
2. **Context-Aware Metadata Filtering:**
   - **Group Filter:** Filter results by one or more WhatsApp study groups.
   - **Sender Filter:** Restrict documents to specific senders (e.g. only materials shared by `Prof. Alan`).
   - **Date Range Filter:** Restrict documents to specific time windows using ISO dates (`YYYY-MM-DD`).
   - **Require Linked Message:** Exclude standalone files imported without WhatsApp conversational context.
   - **Message Keyword Filter:** Require the WhatsApp sharing message or conversation to contain specific keywords (e.g. `midterm`, `quiz`).
3. **Strict Missing Metadata Semantics:**
   - If a filter is active, candidate documents lacking the necessary metadata (e.g. unlinked documents or unknown senders) are excluded with explicit diagnostic reasons (`is_missing_metadata = True`). They are never silently treated as matches.
   - Default behavior: when filters are inactive, all candidates pass through untouched.
4. **Zero Impact on Relevance Scoring:**
   - Conversational mentions do not artificially inflate document content relevance scores. Ranking remains anchored in document chunk text.
5. **Local-First & Privacy Preserving:**
   - All WhatsApp context is queried directly from local SQLite tables. Zero data is sent to external services or Gemini.

---

## WhatsApp Context-Aware Ranking (Phase 4, Step 4.2)

### Secondary Conversational Relevance Signal
While Step 4.1 provides metadata filtering, Step 4.2 introduces context-aware ranking: when candidate documents have comparable content relevance, the teacher's caption (e.g. *"Here is the 3NF and BCNF normalization guide for Unit 3"*) or surrounding discussion serves as a secondary relevance signal to surface the best document.

### Ranking Architecture & Invariants
1. **Content Dominance:**
   Document content relevance from the cross-encoder (`BAAI/bge-reranker-base`) is the primary signal. WhatsApp context is secondary ($W_{\text{context}} = 0.08$). A weak content match can **never** leapfrog a strong content match solely because its caption contains query terms.
2. **Deterministic Lexical Context Scoring:**
   - Evaluates direct sharing message and original filename (75% weight).
   - Evaluates surrounding 5-minute conversation window (25% weight, or discounted to 70% if no caption match).
   - Case-insensitive, punctuation-normalized, and English stopword-filtered token overlap coverage + verbatim multi-word phrase bonus (+0.15).
   - Bounded context score $S_{\text{context}} \in [0.0, 1.0]$.
3. **Scale-Invariant Additive Score Combination:**
   $$\text{Final Score} = S_{\text{content}} + W_{\text{context}} \cdot \max(\Delta_C, 1.0) \cdot S_{\text{context}}$$
   Where $\Delta_C$ is the dynamic content score range across candidates. The boost is strictly capped to $8\%$ of the score range.
4. **Missing Context Neutrality:**
   Unlinked files or documents without WhatsApp context receive $S_{\text{context}} = 0.0$ and $\text{boost} = 0.0$. Their final score equals their content score with zero penalty.
5. **Inspectable Explanations & Transparency:**
   - Preserves original content score and original content rank alongside final score and final rank.
   - Generates factual human-readable explanations (e.g. *"Rank improved from #2 to #1 (+0.240 context bonus from WhatsApp context matching [dbms, normalization])"*).
   - Expandable breakdown of direct message, surrounding context, composite context scores, and matched query terms.
6. **Resilient Fallback:**
   If context scoring fails, the pipeline seamlessly falls back to pure content ranking without crashing.

---

## Running Tests

Execute the automated test suite using `pytest`:

```powershell
pytest -v
```

All 134+ standard unit and integration tests use deterministic mocks, requiring zero model downloads or network calls.

### Running the Real-Model Smoke Test (Optional)
To verify real model weight downloading, caching, and inference using `BAAI/bge-reranker-base`:

```powershell
$env:RUN_REAL_RERANKER_SMOKE_TEST="1"
pytest tests/test_document_ranking.py -k "test_real_model_smoke_test" -v
```

All tests execute against temporary in-memory or isolated filesystem directories (`tmp_path`) without touching `~/.studyrag` or existing user databases.


---

## Launching the Application

Run the Streamlit application:

```powershell
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`, displaying the foundation status dashboard, verifying the SQLite schema, and reporting active configuration settings.
