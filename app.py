"""StudyRAG — Application Entry Point.

Minimal foundation UI verifying configuration, database schema, and environment status.
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import (
    ConfigError,
    Settings,
    ensure_app_dirs,
    get_app_dir,
    get_chroma_dir,
    get_db_path,
    get_log_path,
    get_settings_path,
    load,
)
from core.db import (
    DatabaseError,
    FTS5NotSupportedError,
    get_all_groups,
    get_attachment_ref_counts,
    get_attachment_refs,
    get_existing_tables,
    get_file_by_id,
    get_file_chunks,
    get_group_files,
    get_processing_status_counts,
    get_vector_index_counts,
    init_db,
)
from core.extraction import process_file, process_group_files
from core.vector_store import (
    index_pending_chunks,
    search_chunks_hybrid,
    search_chunks_semantic,
)
from core.ingest import (
    IngestResult,
    get_candidates_for_ref,
    ingest_folder,
    ingest_zip,
    link_attachment_ref_manually,
    reconcile_attachment_refs,
    recover_file_or_folder,
)


# Page Configuration
st.set_page_config(
    page_title="StudyRAG - Foundation",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern premium aesthetic
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.25rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 50%, #1D4ED8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .status-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .badge-ok {
        background-color: #065F46;
        color: #34D399;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-warn {
        background-color: #78350F;
        color: #FBBF24;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .path-text {
        font-family: monospace;
        color: #38BDF8;
        background: #0F172A;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header() -> None:
    """Render top title banner."""
    st.markdown('<div class="main-header">📚 StudyRAG</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Local-first exam intelligence and syllabus-targeted document retrieval</div>',
        unsafe_allow_html=True,
    )


def initialize_foundation() -> tuple[Settings | None, list[str], str | None]:
    """Safely initialize directories, configuration, and database.

    Returns:
        tuple of (settings, table_list, error_message)
    """
    try:
        # 1. Ensure application directory structure outside repo
        ensure_app_dirs()

        # 2. Load settings or fallback to defaults
        settings = load()

        # 3. Initialize SQLite schema & FTS5
        init_db()

        # 4. Read established tables
        tables = get_existing_tables()

        return settings, tables, None
    except ConfigError as err:
        return None, [], f"Configuration Error: {err}"
    except FTS5NotSupportedError as err:
        return None, [], f"SQLite FTS5 Extension Missing: {err}"
    except DatabaseError as err:
        return None, [], f"Database Initialization Error: {err}"


def render_ingestion_preview(settings: Settings) -> None:
    """Render minimal Step 1.2 testable ingestion workflow."""
    st.subheader("📥 Ingestion Pipeline (Step 1.2 Preview)")

    with st.expander("Import WhatsApp Study Group (ZIP or Folder)", expanded=True):
        col_src, col_name = st.columns([1, 1])
        with col_src:
            import_type = st.radio(
                "Source Type",
                ["ZIP Archive (.zip)", "Local Folder"],
                horizontal=True,
            )
        with col_name:
            group_name = st.text_input("Study Group Name", placeholder="e.g. Physics 101")

        target_path_input = st.text_input(
            "Path to ZIP or Folder",
            placeholder=r"C:\Users\username\Downloads\WhatsApp Chat.zip or folder path",
            help="Enter the absolute path to your exported WhatsApp chat ZIP or document folder.",
        )

        uploaded_zip = None
        if import_type.startswith("ZIP"):
            uploaded_zip = st.file_uploader("Or upload export ZIP", type=["zip"])

        if st.button("Start Ingestion", type="primary", use_container_width=True):
            if not group_name.strip():
                st.error("Please enter a valid study group name.")
                return

            result: IngestResult | None = None

            try:
                with st.spinner("Processing documents and chat log..."):
                    if uploaded_zip is not None:
                        temp_zip = Path(get_app_dir()) / "temp_upload.zip"
                        temp_zip.write_bytes(uploaded_zip.getvalue())
                        try:
                            result = ingest_zip(
                                zip_path=temp_zip,
                                group_name=group_name.strip(),
                            )
                        finally:
                            if temp_zip.exists():
                                temp_zip.unlink()
                    elif target_path_input.strip():
                        p = Path(target_path_input.strip()).resolve()
                        if not p.exists():
                            st.error(f"Path does not exist: {p}")
                            return

                        if import_type.startswith("ZIP"):
                            result = ingest_zip(zip_path=p, group_name=group_name.strip())
                        else:
                            result = ingest_folder(folder_path=p, group_name=group_name.strip())
                    else:
                        st.error("Please provide a file/folder path or upload a ZIP file.")
                        return

            except Exception as ex:
                st.error(f"Ingestion failed with exception: {ex}")
                return

            if result:
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Total Found", result.total_found)
                with m2:
                    st.metric("Imported", result.imported_count)
                with m3:
                    st.metric("Duplicates Skipped", result.duplicates_skipped)
                with m4:
                    st.metric("Missing Refs", result.missing_attachments_count)
                with m5:
                    st.metric("Failed", result.failed_count)

                if result.errors:
                    for err in result.errors:
                        st.error(f"Error: {err}")
                elif result.imported_count > 0:
                    st.success(
                        f"Successfully imported {result.imported_count} document(s) for group '{result.group_name}' (ID: {result.group_id})."
                    )
                elif result.duplicates_skipped > 0:
                    st.info(f"All {result.duplicates_skipped} discovered document(s) were already imported.")

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)

                if result.per_file_failures:
                    st.error("Individual file extraction/registration failures:")
                    for fname, reason in result.per_file_failures.items():
                        st.write(f"- **{fname}**: {reason}")

                if result.registered_files:
                    with st.expander(f"View Registered Files ({len(result.registered_files)})", expanded=False):
                        for rf in result.registered_files:
                            st.markdown(f"- `{rf}`")


def render_attachment_recovery(settings: Settings) -> None:
    """Render Step 1.3 Missing Attachment Tracking & Recovery interface."""
    st.subheader("📎 Missing Attachment Tracking & Recovery (Step 1.3)")

    st.warning(
        "⚠️ **WhatsApp Export Advisory:** WhatsApp phone exports often omit media files unless 'Export with Media' is explicitly selected. "
        "StudyRAG tracks all referenced attachments so you can download missing documents manually and import them below."
    )

    all_groups = get_all_groups()
    if not all_groups:
        st.info("No study groups imported yet. Ingest a WhatsApp group above to view and recover attachments.")
        return

    # Filter controls
    f_col1, f_col2 = st.columns([1, 1])
    with f_col1:
        group_options = {"All Groups": None} | {f"{g['name']} (ID: {g['id']})": g["id"] for g in all_groups}
        selected_group_label = st.selectbox("Filter by Study Group", list(group_options.keys()))
        selected_group_id = group_options[selected_group_label]

    with f_col2:
        status_options = ["All", "missing", "available", "ambiguous", "unsupported", "import_failed"]
        selected_status = st.selectbox("Filter by Status", status_options)

    # Summary Metrics from database
    counts = get_attachment_ref_counts(group_id=selected_group_id)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Total References", counts["total"])
    with c2:
        st.metric("Missing", counts["missing"])
    with c3:
        st.metric("Available", counts["available"])
    with c4:
        st.metric("Ambiguous", counts["ambiguous"])
    with c5:
        st.metric("Unsupported", counts["unsupported"])
    with c6:
        st.metric("Import Failed", counts["import_failed"])

    # Reference review table
    refs = get_attachment_refs(group_id=selected_group_id, status=selected_status)

    with st.expander(f"📋 Attachment References List ({len(refs)})", expanded=True):
        if not refs:
            st.write("No attachment references matching the current filter.")
        else:
            for r in refs:
                badge_style = "badge-ok" if r["status"] == "available" else "badge-warn"
                linked_fn = "None"
                if r["file_id"]:
                    f_row = get_file_by_id(r["file_id"])
                    if f_row:
                        linked_fn = f"{f_row['filename']} (ID: {r['file_id']})"

                st.markdown(
                    f"""
                    <div class="status-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <strong>Ref #{r['id']}: {r['original_filename'] or '<em>[Filename Unavailable / Omitted]</em>'}</strong>
                            <span class="{badge_style}">{r['status'].upper()}</span>
                        </div>
                        <p style="margin-top:0.5rem; margin-bottom:0.25rem;">
                            <strong>Sender:</strong> {r['sender'] or 'Unknown'} &nbsp;|&nbsp;
                            <strong>Time:</strong> {r['sent_at'] or 'N/A'} &nbsp;|&nbsp;
                            <strong>Method:</strong> {r['resolution_method'] or 'unresolved'} &nbsp;|&nbsp;
                            <strong>Linked File:</strong> <code>{linked_fn}</code>
                        </p>
                        <p style="margin-bottom:0.25rem;"><strong>Context/Caption:</strong> {r['context'] or r['caption'] or '<em>None</em>'}</p>
                        <p style="color:#94A3B8; font-size:0.85rem; margin-bottom:0;"><strong>Note:</strong> {r['resolution_note'] or 'None'}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Recovery Actions
    st.markdown("#### 🛠️ Recovery & Manual Linking Actions")
    rec_tab1, rec_tab2 = st.tabs(["Import Later Download", "Manual Match & Reconcile"])

    with rec_tab1:
        st.markdown("**Recover Downloaded Document or Folder**")
        target_group_for_rec = st.selectbox(
            "Target Study Group for Recovery",
            [g for g in all_groups],
            format_func=lambda g: f"{g['name']} (ID: {g['id']})",
            key="rec_group_select",
        )
        rec_path_input = st.text_input(
            "File or Folder Path to Recover",
            placeholder=r"C:\Users\username\Downloads\DBMS_Unit_5.pdf",
            key="rec_path_input",
        )
        if st.button("Recover Document(s)", type="primary"):
            if not rec_path_input.strip():
                st.error("Please enter a valid file or folder path.")
            else:
                try:
                    p = Path(rec_path_input.strip()).resolve()
                    if not p.exists():
                        st.error(f"Path does not exist: {p}")
                    else:
                        rec_res = recover_file_or_folder(
                            target_path=p,
                            group_id=target_group_for_rec["id"],
                        )
                        st.success(
                            f"Recovery finished: {len(rec_res.imported_files)} file(s) imported, "
                            f"{len(rec_res.resolved_references)} reference(s) now available, "
                            f"{rec_res.remaining_missing} remaining missing."
                        )
                        if rec_res.warnings:
                            for w in rec_res.warnings:
                                st.warning(w)
                        if rec_res.errors:
                            for e in rec_res.errors:
                                st.error(e)
                        st.rerun()
                except Exception as ex:
                    st.error(f"Recovery failed: {ex}")

    with rec_tab2:
        st.markdown("**Manual Link / Reconcile References**")
        col_m1, col_m2 = st.columns([1, 1])

        unresolved_for_manual = get_attachment_refs(
            group_id=selected_group_id,
            status="missing",
        ) + get_attachment_refs(
            group_id=selected_group_id,
            status="ambiguous",
        )

        with col_m1:
            if not unresolved_for_manual:
                st.info("No missing or ambiguous references in the selected group to link.")
            else:
                chosen_ref = st.selectbox(
                    "Select Unresolved Reference",
                    unresolved_for_manual,
                    format_func=lambda r: f"Ref #{r['id']}: {r['original_filename'] or '[No name]'} ({r['status']})",
                )

                candidates = get_candidates_for_ref(chosen_ref["id"])
                if not candidates:
                    st.warning("No registered files in this group available to link.")
                else:
                    chosen_file = st.selectbox(
                        "Select Registered File to Link",
                        candidates,
                        format_func=lambda f: f"File #{f['id']}: {f['filename']} ({f['path']})",
                    )
                    if st.button("Link Selected File to Reference", type="primary"):
                        ok, msg = link_attachment_ref_manually(chosen_ref["id"], chosen_file["id"])
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

        with col_m2:
            st.markdown("**Retry Auto-Reconciliation**")
            st.write("Scan all registered files in the group and attempt automatic matching for unresolved references.")
            if selected_group_id is not None:
                if st.button("Run Reconcile on Selected Group"):
                    res_counts = reconcile_attachment_refs(group_id=selected_group_id)
                    st.success(
                        f"Reconcile complete: {res_counts['resolved']} newly resolved, "
                        f"{res_counts['ambiguous']} ambiguous, {res_counts['remaining_missing']} still missing."
                    )
                    st.rerun()
            else:
                st.write("Select a specific study group to run auto-reconciliation.")


def render_document_processing(settings: Settings) -> None:
    """Render Step 2.1 Document Text Extraction and Chunking interface."""
    st.subheader("📄 Document Processing & Chunking (Step 2.1)")

    st.markdown(
        "Extract text from registered documents (PDF, PowerPoint, Word) page-by-page, "
        "split into source-aware overlapping chunks, and index into SQLite FTS5 for search."
    )

    all_groups = get_all_groups()
    if not all_groups:
        st.info("No study groups found. Import a WhatsApp group above to begin processing documents.")
        return

    # Filter controls
    group_options = {"All Groups": None} | {f"{g['name']} (ID: {g['id']})": g["id"] for g in all_groups}
    selected_group_label = st.selectbox(
        "Select Study Group for Processing",
        list(group_options.keys()),
        key="doc_proc_group_select",
    )
    selected_group_id = group_options[selected_group_label]

    # Metrics overview from SQLite
    counts = get_processing_status_counts(group_id=selected_group_id)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Total Files", counts["total"])
    with c2:
        st.metric("Pending", counts["new"])
    with c3:
        st.metric("Processed", counts["processed"])
    with c4:
        st.metric("Empty / Scanned", counts["empty"])
    with c5:
        st.metric("Unsupported", counts["unsupported"])
    with c6:
        st.metric("Failed", counts["failed"])

    # Batch action buttons
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("🚀 Process Pending Documents", type="primary", use_container_width=True):
            with st.spinner("Processing pending documents..."):
                batch_res = process_group_files(
                    group_id=selected_group_id,
                    force=False,
                    settings=settings,
                )
                st.success(
                    f"Processing complete: {batch_res.processed_count} processed, "
                    f"{batch_res.empty_count} empty, {batch_res.unsupported_count} unsupported, "
                    f"{batch_res.failed_count} failed out of {batch_res.total_files} pending file(s)."
                )
                st.rerun()

    with col_btn2:
        if st.button("🔄 Reprocess All Documents (Force)", use_container_width=True):
            with st.spinner("Forcing complete re-extraction and chunking..."):
                batch_res = process_group_files(
                    group_id=selected_group_id,
                    force=True,
                    settings=settings,
                )
                st.success(
                    f"Forced re-processing complete: {batch_res.processed_count} processed, "
                    f"{batch_res.empty_count} empty, {batch_res.failed_count} failed."
                )
                st.rerun()

    # Detailed registered files list
    registered_files = get_group_files(group_id=selected_group_id) if selected_group_id else []
    if not selected_group_id:
        # Load all files across all groups
        all_f = []
        for g in all_groups:
            all_f.extend(get_group_files(g["id"]))
        registered_files = all_f

    with st.expander(f"📑 Registered Documents List ({len(registered_files)})", expanded=True):
        if not registered_files:
            st.write("No documents registered in the selected group.")
        else:
            for f in registered_files:
                f_id = f["id"]
                fn = f["filename"]
                ext = f["ext"]
                stat = f["status"]
                err = f["error_message"] if "error_message" in f.keys() else None
                chunk_cnt = f["chunk_count"] if "chunk_count" in f.keys() else 0

                badge = "badge-ok" if stat == "processed" else "badge-warn"
                st.markdown(
                    f"""
                    <div class="status-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <strong>File #{f_id}: {fn}</strong>
                            <span class="{badge}">{stat.upper()}</span>
                        </div>
                        <p style="margin-top:0.4rem; margin-bottom:0.25rem; font-size:0.9rem;">
                            <strong>Type:</strong> <code>{ext}</code> &nbsp;|&nbsp;
                            <strong>Chunks:</strong> <b>{chunk_cnt or 0}</b> &nbsp;|&nbsp;
                            <strong>Sender:</strong> {f['sender'] or 'Unknown'}
                        </p>
                        {f'<p style="color:#EF4444; font-size:0.85rem; margin-bottom:0.25rem;"><strong>Diagnostic:</strong> {err}</p>' if err else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_act1, col_act2 = st.columns([1, 3])
                with col_act1:
                    if st.button(f"Reprocess #{f_id}", key=f"reproc_{f_id}"):
                        res = process_file(file_id=f_id, force=True, settings=settings)
                        if res.status == "processed":
                            st.success(f"Reprocessed: {res.chunk_count} chunk(s) generated.")
                        elif res.status == "empty":
                            st.warning(f"Extracted but empty: {res.error_message}")
                        else:
                            st.error(f"Failed: {res.error_message}")
                        st.rerun()

                with col_act2:
                    if chunk_cnt and chunk_cnt > 0:
                        with st.expander(f"Inspect {chunk_cnt} Extracted Chunks", expanded=False):
                            chunks = get_file_chunks(f_id)
                            for chk in chunks[:10]:  # Show preview of first 10 chunks
                                loc = chk["location"] or f"Page {chk['page']}" if chk["page"] else "Document"
                                st.markdown(f"**Chunk #{chk['chunk_index']} ({loc}):**")
                                st.text(chk["text"][:300] + ("..." if len(chk["text"]) > 300 else ""))
                            if len(chunks) > 10:
                                st.caption(f"... and {len(chunks) - 10} more chunks.")


def render_vector_indexing_and_search(settings: Settings) -> None:
    """Render Step 2.2 Vector Indexing and Hybrid Candidate Retrieval interface."""
    st.subheader("🔍 Vector Indexing & Hybrid Retrieval (Step 2.2)")

    st.markdown(
        "Manage local SentenceTransformers vector embeddings (`BAAI/bge-small-en-v1.5`), "
        "synchronize chunks with local ChromaDB storage, and test unified hybrid search combining lexical and semantic signals."
    )

    all_groups = get_all_groups()
    if not all_groups:
        st.info("No study groups found. Ingest and extract documents above first.")
        return

    # Filter controls
    group_options = {"All Groups": None} | {f"{g['name']} (ID: {g['id']})": g["id"] for g in all_groups}
    selected_group_label = st.selectbox(
        "Select Study Group for Vector Indexing & Search",
        list(group_options.keys()),
        key="vec_group_select",
    )
    selected_group_id = group_options[selected_group_label]

    # Metrics overview from SQLite
    counts = get_vector_index_counts(group_id=selected_group_id, embed_model=settings.embed_model)
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        st.metric("Eligible Chunks", counts["eligible_chunks"])
    with v2:
        st.metric("Indexed in ChromaDB", counts["indexed_chunks"])
    with v3:
        st.metric("Pending Embeddings", counts["pending_chunks"])
    with v4:
        st.metric("Fully Indexed Files", f"{counts['indexed_files']}/{counts['total_files']}")

    # Indexing actions
    col_idx1, col_idx2 = st.columns([1, 1])
    with col_idx1:
        if st.button("⚡ Generate Embeddings & Index Pending Chunks", type="primary", use_container_width=True):
            with st.spinner("Generating embeddings with local model and indexing into ChromaDB..."):
                idx_res = index_pending_chunks(
                    group_id=selected_group_id,
                    embed_model=settings.embed_model,
                )
                if idx_res.errors:
                    for err in idx_res.errors:
                        st.error(err)
                st.success(
                    f"Indexing complete: {idx_res.indexed_count} new chunks indexed, "
                    f"{idx_res.skipped_count} skipped (unchanged), {idx_res.failed_count} failed."
                )
                st.rerun()

    with col_idx2:
        if st.button("🔄 Re-Index Chunks (Force)", use_container_width=True):
            with st.spinner("Re-embedding all eligible chunks into ChromaDB..."):
                idx_res = index_pending_chunks(
                    group_id=selected_group_id,
                    embed_model=settings.embed_model,
                )
                st.success(f"Re-indexing complete: {idx_res.indexed_count} chunks indexed.")
                st.rerun()

    st.markdown("---")
    st.markdown("#### 🎯 Interactive Hybrid Candidate Search Tester")

    q_col1, q_col2, q_col3 = st.columns([2, 1, 1])
    with q_col1:
        query_input = st.text_input(
            "Topic / Search Query",
            placeholder="e.g. database normalization 3NF, concurrency locks, encryption",
            key="hybrid_query_input",
        )
    with q_col2:
        search_mode = st.selectbox(
            "Search Method",
            ["Hybrid (FTS5 + Semantic)", "Semantic Only (ChromaDB)", "Keyword Only (FTS5)"],
            key="hybrid_mode_select",
        )
    with q_col3:
        search_limit = st.slider("Result Limit", min_value=1, max_value=30, value=10, key="hybrid_limit")

    if st.button("Search Candidate Chunks", type="primary"):
        if not query_input.strip():
            st.warning("Please enter a valid search query.")
        else:
            with st.spinner("Retrieving candidate chunks..."):
                results = []
                if search_mode == "Keyword Only (FTS5)":
                    from core.db import search_chunks_fts
                    rows = search_chunks_fts(
                        query=query_input.strip(),
                        group_id=selected_group_id,
                        limit=search_limit,
                    )
                    from core.vector_store import HybridSearchResult
                    for idx, r in enumerate(rows):
                        results.append(
                            HybridSearchResult(
                                chunk_id=str(r["id"]),
                                file_id=int(r["file_id"]),
                                filename=str(r["filename"]),
                                group_id=int(r["group_id"]),
                                text=str(r["text"]),
                                location=str(r["location"] or ""),
                                page=int(r["page"]) if r["page"] is not None else None,
                                matched_by="keyword",
                                keyword_score=1.0 / (1.0 + 0.1 * idx),
                                sender=str(r["sender"]) if "sender" in r.keys() and r["sender"] else None,
                            )
                        )
                elif search_mode == "Semantic Only (ChromaDB)":
                    sem_res = search_chunks_semantic(
                        query=query_input.strip(),
                        top_k=search_limit,
                        group_id=selected_group_id,
                        embed_model=settings.embed_model,
                    )
                    from core.vector_store import HybridSearchResult
                    for s in sem_res:
                        results.append(
                            HybridSearchResult(
                                chunk_id=s.chunk_id,
                                file_id=s.file_id,
                                filename=s.filename,
                                group_id=s.group_id,
                                text=s.text,
                                location=s.location,
                                page=s.page,
                                matched_by="semantic",
                                semantic_distance=s.semantic_distance,
                                semantic_similarity=s.semantic_similarity,
                                sender=s.sender,
                            )
                        )
                else:  # Hybrid
                    results = search_chunks_hybrid(
                        query=query_input.strip(),
                        limit=search_limit,
                        group_id=selected_group_id,
                        embed_model=settings.embed_model,
                    )

                if not results:
                    st.info(f"No matching chunks found for query '{query_input}'.")
                else:
                    st.success(f"Retrieved {len(results)} candidate chunk(s):")
                    for r in results:
                        if r.matched_by == "both":
                            badge_cls = "badge-ok"
                            badge_label = "CONSENSUS (KEYWORD + SEMANTIC)"
                        elif r.matched_by == "semantic":
                            badge_cls = "badge-neutral"
                            badge_label = "SEMANTIC ONLY"
                        else:
                            badge_cls = "badge-warn"
                            badge_label = "KEYWORD ONLY"

                        score_details = []
                        if r.semantic_distance is not None:
                            score_details.append(f"Cosine Dist: <code>{r.semantic_distance:.3f}</code>")
                            if r.semantic_similarity is not None:
                                score_details.append(f"Sim: <code>{r.semantic_similarity * 100:.1f}%</code>")
                        if r.keyword_score is not None:
                            score_details.append(f"KW Rank: <code>{r.keyword_score:.3f}</code>")

                        score_str = " &nbsp;|&nbsp; ".join(score_details)

                        st.markdown(
                            f"""
                            <div class="status-card">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <strong>{r.filename} — {r.location or 'Document'}</strong>
                                    <span class="{badge_cls}">{badge_label}</span>
                                </div>
                                <p style="margin-top:0.4rem; margin-bottom:0.4rem; font-size:0.85rem; color:#94A3B8;">
                                    Chunk ID: <code>{r.chunk_id}</code> &nbsp;|&nbsp; {score_str}
                                    {f"&nbsp;|&nbsp; Sender: <b>{r.sender}</b>" if r.sender else ""}
                                </p>
                                <p style="margin-bottom:0; font-size:0.95rem; white-space:pre-wrap;">{r.text}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


def main() -> None:
    """Application entry point."""
    render_header()

    settings, tables, error = initialize_foundation()

    if error:
        st.error(f"❌ Foundation Initialization Failed:\n\n{error}")
        st.info(
            "Please verify your configuration in `~/.studyrag/settings.json` or check system SQLite FTS5 support."
        )
        return

    assert settings is not None

    # Foundation Status Overview Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="System Status", value="Ready", delta="Step 1.3")
    with col2:
        st.metric(label="Database Tables", value=f"{len(tables)} Active", delta="SQLite + FTS5")
    with col3:
        st.metric(label="Strictness", value=f"{settings.strictness:.2f}")
    with col4:
        st.metric(label="Token Cap", value=f"{settings.daily_token_cap:,}")

    st.markdown("---")

    # Detailed Status Layout
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("📁 Application Storage")
        st.markdown(
            f"""
            <div class="status-card">
                <p><strong>Application Home:</strong> <span class="path-text">{get_app_dir()}</span></p>
                <p><strong>Settings File:</strong> <span class="path-text">{get_settings_path()}</span></p>
                <p><strong>SQLite Database:</strong> <span class="path-text">{get_db_path()}</span></p>
                <p><strong>Vector Chroma Store:</strong> <span class="path-text">{get_chroma_dir()}</span></p>
                <p><strong>Application Log:</strong> <span class="path-text">{get_log_path()}</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("⚙️ Active Configuration")
        with st.expander("View Full Settings JSON", expanded=False):
            st.json(settings.model_dump())

    with right_col:
        st.subheader("🗄️ Database Foundation")
        st.markdown(
            """
            <div class="status-card">
                <p><span class="badge-ok">✓ Active</span> Foreign Key Constraints (<code>PRAGMA foreign_keys = ON</code>)</p>
                <p><span class="badge-ok">✓ Active</span> FTS5 Virtual Table & Auto-sync Triggers</p>
                <p><span class="badge-ok">✓ Active</span> Persistent Attachment Tracking (<code>attachment_refs</code>)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Registered Schema Tables:**")
        table_cols = st.columns(2)
        for i, table_name in enumerate(tables):
            with table_cols[i % 2]:
                st.markdown(f"- ` {table_name} `")

    st.markdown("---")
    render_ingestion_preview(settings)

    st.markdown("---")
    render_attachment_recovery(settings)

    st.markdown("---")
    render_document_processing(settings)

    st.markdown("---")
    render_vector_indexing_and_search(settings)

    st.markdown("---")
    st.info(
        "💡 **Step 2.2 Complete:** Local embeddings, ChromaDB vector indexing, and hybrid candidate retrieval active. "
        "Subsequent steps will implement cross-encoder candidate reranking and Gemini syllabus grading."
    )


if __name__ == "__main__":
    main()


