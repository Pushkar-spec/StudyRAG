"""StudyRAG — Application Entry Point.

Minimal foundation UI verifying configuration, database schema, and environment status.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import streamlit as st

logger = logging.getLogger(__name__)

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
from core.document_ranking import (
    RankedDocument,
    SearchRankedResult,
    search_ranked_documents,
)
from core.whatsapp_context import (
    DocumentWhatsAppContext,
    WhatsAppFilterCriteria,
    get_available_date_range,
    get_available_senders,
    get_whatsapp_context_for_documents,
)
from core.context_ranking import (
    DEFAULT_CONTEXT_WEIGHT,
    WhatsAppContextScore,
    apply_context_ranking,
    score_whatsapp_context,
)
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
    normalize_input_path,
    reconcile_attachment_refs,
    recover_file_or_folder,
    validate_recovery_target,
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
                        try:
                            p = normalize_input_path(target_path_input.strip())
                        except Exception as p_err:
                            st.error(f"Invalid path: {p_err}")
                            return

                        if not p.exists():
                            st.error(f"Path does not exist: '{p}'")
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

    raw_groups = get_all_groups()
    if not raw_groups:
        st.info("No study groups imported yet. Ingest a WhatsApp group above to view and recover attachments.")
        return

    all_groups = [dict(g) for g in raw_groups]
    group_name_map = {g["id"]: g["name"] for g in all_groups}
    group_ids: list[int] = [g["id"] for g in all_groups]

    # Filter controls
    f_col1, f_col2 = st.columns([1, 1])
    with f_col1:
        group_filter_options: list[int | None] = [None] + group_ids
        selected_group_id = st.selectbox(
            "Filter by Study Group",
            options=group_filter_options,
            format_func=lambda gid: "All Groups" if gid is None else f"{group_name_map.get(gid, f'Group #{gid}')} (ID: {gid})",
            key="rec_filter_group_id",
        )

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
    ref_rows = get_attachment_refs(group_id=selected_group_id, status=selected_status)
    refs = [dict(r) for r in ref_rows]

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
                        f_dict = dict(f_row)
                        linked_fn = f"{f_dict['filename']} (ID: {r['file_id']})"

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
            options=group_ids,
            format_func=lambda gid: f"{group_name_map.get(gid, f'Group #{gid}')} (ID: {gid})",
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
                p, val_err = validate_recovery_target(rec_path_input.strip())
                if val_err:
                    st.error(val_err)
                else:
                    try:
                        rec_res = recover_file_or_folder(
                            target_path=p,
                            group_id=target_group_for_rec,
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
                    except Exception as ex:
                        st.error(f"Recovery failed: {ex}")

    with rec_tab2:
        st.markdown("**Manual Link / Reconcile References**")
        col_m1, col_m2 = st.columns([1, 1])

        unresolved_rows = get_attachment_refs(
            group_id=selected_group_id,
            status="missing",
        ) + get_attachment_refs(
            group_id=selected_group_id,
            status="ambiguous",
        )
        unresolved_for_manual = [dict(r) for r in unresolved_rows]

        with col_m1:
            if not unresolved_for_manual:
                st.info("No missing or ambiguous references in the selected group to link.")
            else:
                ref_map = {r["id"]: r for r in unresolved_for_manual}
                chosen_ref_id = st.selectbox(
                    "Select Unresolved Reference",
                    options=[r["id"] for r in unresolved_for_manual],
                    format_func=lambda rid: (
                        f"Ref #{rid}: {ref_map[rid]['original_filename'] or '[No name]'} ({ref_map[rid]['status']})"
                        if rid in ref_map else f"Ref #{rid}"
                    ),
                    key="manual_link_ref_select",
                )

                candidate_rows = get_candidates_for_ref(chosen_ref_id)
                candidates = [dict(f) for f in candidate_rows]
                if not candidates:
                    st.warning("No registered files in this group available to link.")
                else:
                    cand_map = {f["id"]: f for f in candidates}
                    chosen_file_id = st.selectbox(
                        "Select Registered File to Link",
                        options=[f["id"] for f in candidates],
                        format_func=lambda fid: (
                            f"File #{fid}: {cand_map[fid]['filename']} ({cand_map[fid]['path']})"
                            if fid in cand_map else f"File #{fid}"
                        ),
                        key="manual_link_file_select",
                    )
                    if st.button("Link Selected File to Reference", type="primary"):
                        ok, msg = link_attachment_ref_manually(chosen_ref_id, chosen_file_id)
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

    raw_groups = get_all_groups()
    if not raw_groups:
        st.info("No study groups found. Import a WhatsApp group above to begin processing documents.")
        return

    all_groups = [dict(g) for g in raw_groups]
    group_name_map = {g["id"]: g["name"] for g in all_groups}
    group_filter_options: list[int | None] = [None] + [g["id"] for g in all_groups]

    # Filter controls
    selected_group_id = st.selectbox(
        "Select Study Group for Processing",
        options=group_filter_options,
        format_func=lambda gid: "All Groups" if gid is None else f"{group_name_map.get(gid, f'Group #{gid}')} (ID: {gid})",
        key="doc_proc_group_select",
    )

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
    raw_files = get_group_files(group_id=selected_group_id) if selected_group_id else []
    if not selected_group_id:
        # Load all files across all groups
        all_f = []
        for g in all_groups:
            all_f.extend(get_group_files(g["id"]))
        raw_files = all_f
    registered_files = [dict(f) for f in raw_files]

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

    raw_groups = get_all_groups()
    if not raw_groups:
        st.info("No study groups found. Ingest and extract documents above first.")
        return

    all_groups = [dict(g) for g in raw_groups]
    group_name_map = {g["id"]: g["name"] for g in all_groups}
    group_filter_options: list[int | None] = [None] + [g["id"] for g in all_groups]

    # Filter controls
    selected_group_id = st.selectbox(
        "Select Study Group for Vector Indexing & Search",
        options=group_filter_options,
        format_func=lambda gid: "All Groups" if gid is None else f"{group_name_map.get(gid, f'Group #{gid}')} (ID: {gid})",
        key="vec_group_select",
    )

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
    st.markdown("#### 🎯 Interactive Document Search & Cross-Encoder Reranker")

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
            [
                "Ranked Documents (Hybrid + Cross-Encoder)",
                "Hybrid Candidates (FTS5 + Semantic)",
                "Semantic Only (ChromaDB)",
                "Keyword Only (FTS5)",
            ],
            key="hybrid_mode_select",
        )
    with q_col3:
        search_limit = st.slider("Result Limit", min_value=1, max_value=30, value=10, key="hybrid_limit")

    # WhatsApp Context & Metadata Filters (Step 4.1)
    with st.expander("💬 WhatsApp Context & Metadata Filters (Optional)", expanded=False):
        wf_col1, wf_col2 = st.columns(2)
        with wf_col1:
            avail_senders = get_available_senders(group_id=selected_group_id)
            selected_senders = st.multiselect(
                "Filter by Sender(s)",
                options=avail_senders,
                key="wa_filter_senders",
                help="Only include documents shared or referenced by selected WhatsApp senders.",
            )
            require_linked = st.checkbox(
                "Require linked WhatsApp sharing message",
                value=False,
                key="wa_filter_require_linked",
                help="Exclude standalone files imported without WhatsApp chat context.",
            )
            enable_context_ranking = st.checkbox(
                "Enable WhatsApp Context-Aware Ranking",
                value=not getattr(settings, "disable_context_ranking", False),
                key="wa_enable_context_ranking",
                help="Conservatively boost document rank when query terms match WhatsApp sharing message or conversation.",
            )
        with wf_col2:
            d_min, d_max = get_available_date_range(group_id=selected_group_id)
            df_col1, df_col2 = st.columns(2)
            with df_col1:
                date_from_val = st.text_input(
                    "Date From (YYYY-MM-DD)",
                    placeholder=d_min[:10] if d_min else "YYYY-MM-DD",
                    key="wa_filter_date_from",
                )
            with df_col2:
                date_to_val = st.text_input(
                    "Date To (YYYY-MM-DD)",
                    placeholder=d_max[:10] if d_max else "YYYY-MM-DD",
                    key="wa_filter_date_to",
                )
            msg_query_val = st.text_input(
                "Filter by Message Keyword",
                placeholder="e.g. quiz, assignment, homework",
                key="wa_filter_msg_query",
                help="Only include documents whose WhatsApp message or surrounding conversation contains this text.",
            )

    # Build criteria object
    whatsapp_filters = WhatsAppFilterCriteria(
        group_ids=[selected_group_id] if selected_group_id is not None else None,
        senders=selected_senders if selected_senders else None,
        date_from=date_from_val.strip() if date_from_val.strip() else None,
        date_to=date_to_val.strip() if date_to_val.strip() else None,
        has_linked_message=True if require_linked else None,
        message_query=msg_query_val.strip() if msg_query_val.strip() else None,
    )

    if search_mode == "Ranked Documents (Hybrid + Cross-Encoder)":
        btn_label = "Search & Rerank Documents"
    else:
        btn_label = "Search Candidate Chunks"

    if st.button(btn_label, type="primary"):
        if not query_input.strip():
            st.warning("Please enter a valid search query.")
        else:
            if search_mode == "Ranked Documents (Hybrid + Cross-Encoder)":
                with st.spinner("Executing hybrid search, cross-encoder reranking, and context retrieval..."):
                    rank_res = search_ranked_documents(
                        query=query_input.strip(),
                        limit=search_limit,
                        group_id=selected_group_id,
                        settings=settings,
                        whatsapp_filters=whatsapp_filters,
                        disable_context_ranking=not enable_context_ranking,
                    )

                if not rank_res.documents:
                    filter_msg = " matching the active WhatsApp context filters." if whatsapp_filters.is_active() else "."
                    st.info(f"No matching documents found for query '{query_input}'{filter_msg}")
                else:
                    active_rerank_model = getattr(settings, "rerank_model", "BAAI/bge-reranker-base")
                    filter_note = f" ({rank_res.excluded_documents_count} excluded by WhatsApp filters)" if rank_res.excluded_documents_count > 0 else ""
                    ctx_note = f" (Context-Aware Ranking active: {rank_res.context_adjustments_count} adjustments)" if rank_res.is_context_ranked else ""
                    if rank_res.is_reranked:
                        st.success(
                            f"Retrieved and reranked {len(rank_res.documents)} document(s) from "
                            f"{rank_res.total_candidates} hybrid candidate chunks using '{active_rerank_model}'{filter_note}{ctx_note}:"
                        )
                    else:
                        st.warning(
                            f"Retrieved {len(rank_res.documents)} document(s) using fallback candidate ranking "
                            f"(cross-encoder unavailable or bypassed){filter_note}{ctx_note}:"
                        )

                    for doc in rank_res.documents:
                        badge_cls = "badge-ok" if doc.is_reranked else "badge-warn"
                        badge_label = "✓ RERANKED" if doc.is_reranked else "FALLBACK RANK"

                        loc_str = ", ".join(doc.source_locations) if doc.source_locations else "Document Body"

                        ctx = getattr(doc, "whatsapp_context", None)
                        wa_badge = ""
                        wa_info_html = ""
                        if ctx and ctx.has_linked_message:
                            wa_badge = '<span class="badge-ok" style="margin-left:0.5rem; font-size:0.75rem;">💬 WhatsApp Linked</span>'
                            sender_txt = f"👤 Sender: <b>{ctx.sender or 'Unknown'}</b>"
                            date_txt = f"📅 Date: <b>{ctx.sent_at[:10]}</b>" if ctx.sent_at else ""
                            wa_info_html = f"""
                            <p style="margin-top:0.2rem; margin-bottom:0.4rem; font-size:0.85rem; color:#A7F3D0;">
                                {sender_txt} {f"&nbsp;|&nbsp; {date_txt}" if date_txt else ""}
                            </p>
                            """
                        else:
                            wa_info_html = """
                            <p style="margin-top:0.2rem; margin-bottom:0.4rem; font-size:0.85rem; color:#94A3B8;">
                                <i>No linked WhatsApp chat message (unlinked file)</i>
                            </p>
                            """

                        rank_title = f"Rank #{doc.rank}: {doc.filename}"
                        if doc.content_rank is not None and doc.content_rank != doc.rank:
                            rank_title = f"Rank #{doc.rank} (Content Rank: #{doc.content_rank}): {doc.filename}"

                        if doc.content_score is not None and abs(doc.document_score - doc.content_score) > 0.0001:
                            score_html = f"Adjusted Score: <code>{doc.document_score:.3f}</code> &nbsp;|&nbsp; Content Score: <code>{doc.content_score:.3f}</code>"
                        else:
                            score_html = f"Document Model Score: <code>{doc.document_score:.3f}</code>"

                        expl_html = ""
                        if getattr(doc, "adjustment_explanation", ""):
                            expl_html = f"""
                            <p style="margin-top:0.2rem; margin-bottom:0.3rem; font-size:0.85rem; color:#FDE047;">
                                💡 {doc.adjustment_explanation}
                            </p>
                            """

                        st.markdown(
                            f"""
                            <div class="status-card">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <strong>{rank_title}</strong>
                                    <div>{wa_badge}<span class="{badge_cls}" style="margin-left:0.5rem;">{badge_label}</span></div>
                                </div>
                                <p style="margin-top:0.4rem; margin-bottom:0.2rem; font-size:0.85rem; color:#94A3B8;">
                                    Group: <b>{doc.group_name}</b> &nbsp;|&nbsp;
                                    Type: <code>{doc.doc_type}</code> &nbsp;|&nbsp;
                                    Distinct Chunks: <b>{doc.chunk_count}</b> &nbsp;|&nbsp;
                                    {score_html}
                                </p>
                                {wa_info_html}
                                {expl_html}
                                <p style="margin-bottom:0.4rem; font-size:0.85rem; color:#60A5FA;">
                                    📍 Source Locations: <b>{loc_str}</b>
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if doc.excerpts:
                            for exc in doc.excerpts[:2]:
                                st.markdown(f"> *\"{exc}\"*")

                        # WhatsApp Context inspection expander
                        if ctx and ctx.has_linked_message:
                            with st.expander(f"💬 Inspect WhatsApp Sharing Message & Context ({len(ctx.all_messages)} msg)", expanded=False):
                                if getattr(doc, "context_score", None):
                                    cs = doc.context_score
                                    matched_terms_str = ", ".join(cs.matched_terms) if cs.matched_terms else "None"
                                    st.markdown(
                                        f"**WhatsApp Relevance Breakdown:** &nbsp;|&nbsp; "
                                        f"Direct Message: <code>{cs.direct_score:.2f}</code> &nbsp;|&nbsp; "
                                        f"Surrounding Context: <code>{cs.surrounding_score:.2f}</code> &nbsp;|&nbsp; "
                                        f"Composite Context: <code>{cs.combined_score:.2f}</code> &nbsp;|&nbsp; "
                                        f"Matched Terms: <code>{matched_terms_str}</code>"
                                    )
                                    st.markdown("---")
                                if ctx.message_text:
                                    st.markdown(f"**Direct Sharing Message:**\n> {ctx.message_text}")
                                if ctx.surrounding_context and ctx.surrounding_context.strip() != (ctx.message_text or "").strip():
                                    st.markdown(f"**Surrounding Conversation (5-min window):**\n```text\n{ctx.surrounding_context}\n```")
                                p_msg = ctx.primary_message
                                orig_fn = p_msg.original_filename if p_msg else doc.filename
                                status_txt = p_msg.status if p_msg else "available"
                                res_meth = p_msg.resolution_method if p_msg else "direct"
                                st.caption(f"Attachment Reference: `{orig_fn}` | Status: `{status_txt}` | Resolution: `{res_meth}`")

                        with st.expander(f"🔍 Supporting Chunks & Relevance Signals ({len(doc.supporting_chunks)})", expanded=False):
                            for idx, c in enumerate(doc.supporting_chunks, start=1):
                                cid = getattr(c, "chunk_id", None) or (c.get("chunk_id") if isinstance(c, dict) else f"chunk_{idx}")
                                loc = getattr(c, "location", None) or (c.get("location") if isinstance(c, dict) else "")
                                matched_by = getattr(c, "matched_by", None) or (c.get("matched_by") if isinstance(c, dict) else "unknown")
                                r_score = getattr(c, "rerank_score", None) or (c.get("rerank_score") if isinstance(c, dict) else None)
                                kw_score = getattr(c, "keyword_score", None) or (c.get("keyword_score") if isinstance(c, dict) else None)
                                sem_sim = getattr(c, "semantic_similarity", None) or (c.get("semantic_similarity") if isinstance(c, dict) else None)
                                text = getattr(c, "text", "") or (c.get("text", "") if isinstance(c, dict) else "")

                                signal_parts = [f"Matched By: <code>{matched_by}</code>"]
                                if r_score is not None:
                                    signal_parts.append(f"Reranker Score: <code>{r_score:.3f}</code>")
                                if sem_sim is not None:
                                    signal_parts.append(f"Semantic Sim: <code>{sem_sim*100:.1f}%</code>")
                                if kw_score is not None:
                                    signal_parts.append(f"KW Rank: <code>{kw_score:.3f}</code>")

                                st.markdown(
                                    f"**Chunk #{idx}** (`{cid}` — *{loc or 'Body'}*) &nbsp;|&nbsp; " + " &nbsp;|&nbsp; ".join(signal_parts)
                                )
                                st.text(text)
                                st.markdown("---")

                    with st.expander("🛠️ Retrieval & Context Pipeline Diagnostics", expanded=False):
                        for diag in rank_res.diagnostics:
                            st.write(f"- {diag}")
                        if rank_res.applied_filters:
                            st.write("**Applied Filters:**")
                            st.json(rank_res.applied_filters)
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
                    else:  # Hybrid Candidates
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
    try:
        render_ingestion_preview(settings)
    except Exception as err:
        st.error(f"⚠️ Ingestion Pipeline section encountered an unexpected error: {err}")
        logger.exception("Failed to render Ingestion Pipeline: %s", err)

    st.markdown("---")
    try:
        render_attachment_recovery(settings)
    except Exception as err:
        st.error(f"⚠️ Attachment Recovery section encountered an unexpected error: {err}")
        logger.exception("Failed to render Attachment Recovery: %s", err)

    st.markdown("---")
    try:
        render_document_processing(settings)
    except Exception as err:
        st.error(f"⚠️ Document Processing section encountered an unexpected error: {err}")
        logger.exception("Failed to render Document Processing: %s", err)

    st.markdown("---")
    try:
        render_vector_indexing_and_search(settings)
    except Exception as err:
        st.error(f"⚠️ Vector Indexing & Semantic Search section encountered an unexpected error: {err}")
        logger.exception("Failed to render Vector Indexing & Semantic Search: %s", err)

    st.markdown("---")
    st.info(
        "💡 **Step 4.2 Complete:** WhatsApp context-aware ranking active on top of hybrid cross-encoder retrieval. "
        "Context relevance conservatively resolves close results while strictly preserving content dominance."
    )


if __name__ == "__main__":
    main()


