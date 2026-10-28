"""Tests for StudyRAG WhatsApp Context-Aware Ranking (Phase 4, Step 4.2).

Validates:
- Lexical context relevance scoring (direct sharing message vs surrounding context).
- Punctuation, case-insensitivity, stopwords, and duplicate-token handling.
- Bounded context scores [0.0, 1.0].
- Missing context neutrality (no penalty for unlinked files).
- Conservative score combination and dominance of content relevance over weak matches.
- Resolving close results for comparable documents.
- Deterministic tie-breaking and preservation of original scores and ranks.
- Precedence of metadata filters over context scoring.
- Resilient fallback when context retrieval or ranking fails.
- Section 9 end-to-end acceptance scenario.
"""

from pathlib import Path
import pytest

from core.context_ranking import (
    DEFAULT_CONTEXT_WEIGHT,
    ContextRankingAdjustment,
    WhatsAppContextScore,
    apply_context_ranking,
    combine_relevance_scores,
    extract_query_tokens,
    score_text_overlap,
    score_whatsapp_context,
)
from core.db import (
    db_session,
    get_or_create_group,
    init_db,
    insert_file,
    upsert_attachment_ref,
)
from core.document_ranking import (
    RankedDocument,
    SearchRankedResult,
    search_ranked_documents,
)
from core.whatsapp_context import (
    DocumentWhatsAppContext,
    WhatsAppFilterCriteria,
    WhatsAppMessageContext,
)


# ---------------------------------------------------------------------------
# Unit Tests: Tokenization and Overlap Scoring
# ---------------------------------------------------------------------------

def test_extract_query_tokens_normalization_and_stopwords():
    """Verify that tokenization handles case, punctuation, stopwords, and whitespace."""
    tokens = extract_query_tokens("  Notes on   DBMS: Normalization & 3NF!  ")
    assert tokens == ["notes", "dbms", "normalization", "3nf"]

    # Stopwords are removed
    assert "on" not in tokens

    # Duplicate words in query are deduplicated preserving order
    tokens_dup = extract_query_tokens("database normalization normalization database")
    assert tokens_dup == ["database", "normalization"]

    # When all words are stopwords, fall back to valid alphanumeric tokens
    tokens_stopwords = extract_query_tokens("this and that")
    assert len(tokens_stopwords) > 0


def test_score_text_overlap_calculation_and_phrase_bonus():
    """Verify lexical overlap coverage and phrase matching."""
    tokens = ["dbms", "normalization"]
    phrase = "dbms normalization"

    # Perfect match + phrase match
    score, matched = score_text_overlap(tokens, phrase, "Here are the DBMS Normalization lecture notes")
    assert matched == ["dbms", "normalization"]
    assert score == 1.0  # 0.85 * 1.0 + 0.15 = 1.0

    # Partial match (1 of 2 tokens, no phrase)
    score_part, matched_part = score_text_overlap(tokens, phrase, "Notes covering basic DBMS fundamentals")
    assert matched_part == ["dbms"]
    assert 0.40 <= score_part <= 0.45  # 0.85 * 0.5 = 0.425

    # Zero match
    score_none, matched_none = score_text_overlap(tokens, phrase, "History of Art and Architecture")
    assert score_none == 0.0
    assert matched_none == []

    # Empty text or tokens
    assert score_text_overlap([], phrase, "Some text")[0] == 0.0
    assert score_text_overlap(tokens, phrase, "")[0] == 0.0
    assert score_text_overlap(tokens, phrase, None)[0] == 0.0


# ---------------------------------------------------------------------------
# Unit Tests: Context Relevance Scoring
# ---------------------------------------------------------------------------

def test_score_whatsapp_context_direct_vs_surrounding():
    """Verify separation of direct sharing-message and surrounding conversation scores."""
    # 1. Document with direct caption
    ctx_direct = DocumentWhatsAppContext(
        file_id=1,
        filename="dbms_unit3.pdf",
        group_id=10,
        group_name="CS101",
        has_linked_message=True,
        primary_message=WhatsAppMessageContext(
            ref_id=1,
            sender="Prof. Alan",
            sent_at="2026-03-20T10:00:00",
            message_text="DBMS Normalization slides for today",
            surrounding_context=None,
            group_id=10,
            group_name="CS101",
            original_filename="dbms_unit3.pdf",
        ),
    )
    score_d = score_whatsapp_context("DBMS normalization", ctx_direct)
    assert score_d.has_context is True
    assert score_d.direct_score > 0.8
    assert score_d.surrounding_score == 0.0
    assert score_d.combined_score == score_d.direct_score

    # 2. Document with only surrounding conversation
    ctx_surr = DocumentWhatsAppContext(
        file_id=2,
        filename="doc2.pdf",
        group_id=10,
        group_name="CS101",
        has_linked_message=True,
        primary_message=WhatsAppMessageContext(
            ref_id=2,
            sender="Prof. Alan",
            sent_at="2026-03-20T10:00:00",
            message_text="Uploading file",
            surrounding_context="[10:00] Prof. Alan: Does everyone understand DBMS normalization?",
            group_id=10,
            group_name="CS101",
            original_filename="doc2.pdf",
        ),
    )
    score_s = score_whatsapp_context("DBMS normalization", ctx_surr)
    assert score_s.direct_score == 0.0
    assert score_s.surrounding_score > 0.8
    # Surrounding only is discounted (0.70 factor)
    assert score_s.combined_score < score_d.combined_score
    assert score_s.combined_score == pytest.approx(0.70 * score_s.surrounding_score, abs=0.01)


def test_score_whatsapp_context_missing_and_unlinked_neutrality():
    """Missing or unlinked context must produce a neutral 0.0 score, never a negative penalty."""
    # None context
    score_none = score_whatsapp_context("DBMS normalization", None)
    assert score_none.has_context is False
    assert score_none.combined_score == 0.0
    assert score_none.direct_score == 0.0
    assert score_none.surrounding_score == 0.0
    assert "no linked" in score_none.explanation.lower()

    # Unlinked document
    ctx_unlinked = DocumentWhatsAppContext(
        file_id=3,
        filename="standalone.pdf",
        group_id=10,
        group_name="CS101",
        has_linked_message=False,
    )
    score_unlinked = score_whatsapp_context("DBMS normalization", ctx_unlinked)
    assert score_unlinked.has_context is False
    assert score_unlinked.combined_score == 0.0


# ---------------------------------------------------------------------------
# Unit Tests: Score Combination & Ranking Adjustment
# ---------------------------------------------------------------------------

def test_combine_relevance_scores_properties():
    """Verify formula guarantees: content dominance, scale invariance, and neutrality."""
    # 1. Missing context produces zero boost and preserves content score exactly
    final_score, boost = combine_relevance_scores(content_score=4.5, context_score=0.0, score_range=3.0)
    assert boost == 0.0
    assert final_score == 4.5

    # 2. Positive context provides bounded boost
    final_boosted, boost_val = combine_relevance_scores(
        content_score=4.5, context_score=1.0, score_range=3.0, weight=0.08
    )
    assert boost_val == pytest.approx(0.08 * 3.0 * 1.0, abs=0.001)  # 0.24
    assert final_boosted == pytest.approx(4.5 + 0.24, abs=0.001)

    # 3. Minimum effective range is 1.0 even if candidate score range is 0.0 (e.g. single document)
    _, boost_zero_range = combine_relevance_scores(content_score=2.0, context_score=1.0, score_range=0.0, weight=0.08)
    assert boost_zero_range == pytest.approx(0.08 * 1.0 * 1.0, abs=0.001)


def test_content_dominance_over_weak_match_with_perfect_caption():
    """A weak content match must NEVER overtake a strong content match solely due to WhatsApp caption."""
    # Document 1: Strong content relevance (4.50), no WhatsApp context
    doc1 = RankedDocument(
        file_id=1,
        filename="strong_doc.pdf",
        doc_type=".pdf",
        group_id=1,
        group_name="CS101",
        document_score=4.50,
        rank=1,
        whatsapp_context=None,
    )

    # Document 2: Weak content relevance (1.00), perfect WhatsApp caption match (1.0)
    ctx_weak = DocumentWhatsAppContext(
        file_id=2,
        filename="weak_doc.pdf",
        group_id=1,
        group_name="CS101",
        has_linked_message=True,
        primary_message=WhatsAppMessageContext(
            ref_id=2,
            sender="Prof. Alan",
            sent_at="2026-03-20T10:00:00",
            message_text="DBMS Normalization lecture slides",
            surrounding_context="DBMS Normalization lecture slides",
            group_id=1,
            group_name="CS101",
            original_filename="weak_doc.pdf",
        ),
    )
    doc2 = RankedDocument(
        file_id=2,
        filename="weak_doc.pdf",
        doc_type=".pdf",
        group_id=1,
        group_name="CS101",
        document_score=1.00,
        rank=2,
        whatsapp_context=ctx_weak,
    )

    reordered, adjustments = apply_context_ranking(
        documents=[doc1, doc2],
        query="DBMS normalization",
        weight=DEFAULT_CONTEXT_WEIGHT,
    )

    # Doc 1 must remain Rank #1
    assert reordered[0].file_id == 1
    assert reordered[0].rank == 1
    # Doc 2 must remain Rank #2 despite 100% caption match
    assert reordered[1].file_id == 2
    assert reordered[1].rank == 2

    # Verify score gap
    assert reordered[0].document_score > reordered[1].document_score + 2.5


def test_close_candidates_reranking_by_context():
    """When two candidate documents have very close content relevance, context resolves the close call."""
    # Document A: content score 3.20, no matching context
    doc_a = RankedDocument(
        file_id=10,
        filename="notes_a.pdf",
        doc_type=".pdf",
        group_id=1,
        group_name="CS101",
        document_score=3.20,
        rank=1,
        whatsapp_context=None,
    )

    # Document B: content score 3.16 (very close, only 0.04 deficit), strong matching caption
    ctx_b = DocumentWhatsAppContext(
        file_id=11,
        filename="notes_b.pdf",
        group_id=1,
        group_name="CS101",
        has_linked_message=True,
        primary_message=WhatsAppMessageContext(
            ref_id=11,
            sender="Prof. Alan",
            sent_at="2026-03-20T10:00:00",
            message_text="DBMS Normalization reference guide",
            surrounding_context=None,
            group_id=1,
            group_name="CS101",
            original_filename="notes_b.pdf",
        ),
    )
    doc_b = RankedDocument(
        file_id=11,
        filename="notes_b.pdf",
        doc_type=".pdf",
        group_id=1,
        group_name="CS101",
        document_score=3.16,
        rank=2,
        whatsapp_context=ctx_b,
    )

    reordered, adjustments = apply_context_ranking(
        documents=[doc_a, doc_b],
        query="DBMS normalization",
        weight=DEFAULT_CONTEXT_WEIGHT,
    )

    # Doc B receives context boost and overtakes Doc A
    assert reordered[0].file_id == 11
    assert reordered[0].rank == 1
    assert reordered[0].content_rank == 2
    assert "Rank improved from #2 to #1" in reordered[0].adjustment_explanation

    assert reordered[1].file_id == 10
    assert reordered[1].rank == 2
    assert reordered[1].content_rank == 1


def test_deterministic_tie_breaking():
    """When final scores are tied, tie-breaking favors content score, chunk count, then file_id."""
    doc1 = RankedDocument(
        file_id=101,
        filename="doc101.pdf",
        doc_type=".pdf",
        group_id=1,
        group_name="CS101",
        document_score=3.0,
        rank=1,
        chunk_count=5,
    )
    doc2 = RankedDocument(
        file_id=102,
        filename="doc102.pdf",
        doc_type=".pdf",
        group_id=1,
        group_name="CS101",
        document_score=3.0,
        rank=2,
        chunk_count=3,
    )

    # Both documents have identical score, Doc 1 has more chunks -> Doc 1 is #1
    reordered, _ = apply_context_ranking([doc2, doc1], query="test query", weight=0.08)
    assert reordered[0].file_id == 101
    assert reordered[1].file_id == 102


def test_disable_context_ranking_preserves_pure_content():
    """Disabling context ranking preserves original ranks and document scores."""
    doc = RankedDocument(
        file_id=1,
        filename="doc1.pdf",
        doc_type=".pdf",
        group_id=1,
        group_name="CS101",
        document_score=3.5,
        rank=1,
    )
    reordered, adjustments = apply_context_ranking(
        [doc], query="test", weight=0.08, disable=True
    )
    assert reordered[0].rank == 1
    assert reordered[0].document_score == 3.5
    assert "context ranking disabled" in reordered[0].adjustment_explanation


# ---------------------------------------------------------------------------
# Section 9 End-to-End Acceptance Scenario
# ---------------------------------------------------------------------------

def test_section_9_end_to_end_acceptance_scenario(tmp_path: Path):
    """Synthetic end-to-end acceptance scenario for query 'DBMS normalization'.

    Scenario:
    - Doc 1 (Strong content 4.0 + matching WhatsApp caption) -> Rank #1.
    - Doc 2 (Strong content 3.9 + no WhatsApp context) -> Rank #2 (unpenalized).
    - Doc 3 (Moderate content 2.5 + moderate surrounding context) -> Rank #3.
    - Doc 4 (Weak content 0.5 + perfect caption match) -> Rank #4 (cannot leapfrog).
    """
    db_path = tmp_path / "studyrag.db"
    init_db(db_path)

    gid = get_or_create_group("CS_DBMS", db_path=db_path)

    # Insert files into SQLite
    f1 = insert_file(gid, "h1", "f1_strong_match.pdf", str(tmp_path / "f1.pdf"), ".pdf", db_path=db_path)
    f2 = insert_file(gid, "h2", "f2_strong_no_ctx.pdf", str(tmp_path / "f2.pdf"), ".pdf", db_path=db_path)
    f3 = insert_file(gid, "h3", "f3_mod_surr.pdf", str(tmp_path / "f3.pdf"), ".pdf", db_path=db_path)
    f4 = insert_file(gid, "h4", "f4_weak_caption.pdf", str(tmp_path / "f4.pdf"), ".pdf", db_path=db_path)

    # Link WhatsApp references
    # Doc 1: Strong matching caption
    upsert_attachment_ref(
        group_id=gid,
        dedup_key="k1",
        status="available",
        file_id=f1,
        sender="Prof. Alan",
        sent_at="2026-03-25T09:00:00",
        caption="Unit 3 DBMS Normalization Lecture Notes",
        context="[08:58] Prof. Alan: Today we cover 3NF and BCNF.\n[09:00] Prof. Alan: Unit 3 DBMS Normalization Lecture Notes",
        original_filename="f1_strong_match.pdf",
        db_path=db_path,
    )

    # Doc 2: No WhatsApp reference (unlinked standalone document)

    # Doc 3: Moderate surrounding context
    upsert_attachment_ref(
        group_id=gid,
        dedup_key="k3",
        status="available",
        file_id=f3,
        sender="Prof. Alan",
        sent_at="2026-03-22T14:00:00",
        caption="Study Guide",
        context="[13:55] Student: Are we tested on normalization?\n[13:58] Prof. Alan: Yes, review 3NF.\n[14:00] Prof. Alan: Study Guide",
        original_filename="f3_mod_surr.pdf",
        db_path=db_path,
    )

    # Doc 4: Weak document with perfect caption
    upsert_attachment_ref(
        group_id=gid,
        dedup_key="k4",
        status="available",
        file_id=f4,
        sender="Student Bob",
        sent_at="2026-03-26T18:00:00",
        caption="DBMS Normalization homework answers",
        context="[18:00] Student Bob: DBMS Normalization homework answers",
        original_filename="f4_weak_caption.pdf",
        db_path=db_path,
    )

    # Insert candidate chunks
    with db_session(db_path) as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO chunks (id, file_id, text, location) VALUES (?, ?, ?, ?)",
                    ("c1", f1, "DBMS normalization rules: 1NF, 2NF, 3NF, and BCNF dependencies.", "Page 1"))
        cur.execute("INSERT INTO chunks (id, file_id, text, location) VALUES (?, ?, ?, ?)",
                    ("c2", f2, "DBMS normalization: functional dependencies and lossless decomposition.", "Page 5"))
        cur.execute("INSERT INTO chunks (id, file_id, text, location) VALUES (?, ?, ?, ?)",
                    ("c3", f3, "DBMS normalization: general overview of relational database topics.", "Page 2"))
        cur.execute("INSERT INTO chunks (id, file_id, text, location) VALUES (?, ?, ?, ?)",
                    ("c4", f4, "DBMS normalization: brief syllabus mention in introductory course.", "Page 1"))

    # Custom mock reranker simulating content relevance scores:
    # f1 -> 4.00, f2 -> 3.90, f3 -> 2.50, f4 -> 0.50
    def mock_custom_reranker(pairs: list[tuple[str, str]]) -> list[float]:
        results = []
        for q, passage in pairs:
            if "BCNF dependencies" in passage:
                results.append(4.00)
            elif "lossless decomposition" in passage:
                results.append(3.90)
            elif "relational database" in passage:
                results.append(2.50)
            else:
                results.append(0.50)
        return results

    # Execute search_ranked_documents
    res = search_ranked_documents(
        query="DBMS normalization",
        limit=10,
        db_path=db_path,
        custom_reranker=mock_custom_reranker,
        custom_embedder=lambda texts: [[0.1] * 384 for _ in texts],
    )

    assert len(res.documents) == 4
    assert res.is_context_ranked is True

    # 1. Verify Ordering
    doc_ranks = {doc.file_id: doc.rank for doc in res.documents}
    assert doc_ranks[f1] == 1, "Doc 1 (strong content + context) must be Rank 1"
    assert doc_ranks[f2] == 2, "Doc 2 (strong content, no context) must be Rank 2"
    assert doc_ranks[f3] == 3, "Doc 3 (moderate content + surrounding context) must be Rank 3"
    assert doc_ranks[f4] == 4, "Doc 4 (weak content + strong caption) must remain at Rank 4"

    # 2. Content relevance dominance: Doc 4 could not overtake Doc 1, Doc 2, or Doc 3
    doc4_res = next(d for d in res.documents if d.file_id == f4)
    doc2_res = next(d for d in res.documents if d.file_id == f2)
    assert doc4_res.document_score < doc2_res.document_score

    # 3. Missing context neutrality: Doc 2 had no context, but suffered zero penalty
    assert doc2_res.whatsapp_context is None or not doc2_res.whatsapp_context.has_linked_message
    assert doc2_res.document_score == pytest.approx(doc2_res.content_score, abs=0.001)

    # 4. Explanations are inspectable
    doc1_res = next(d for d in res.documents if d.file_id == f1)
    assert doc1_res.context_score.combined_score > 0.8
    assert "DBMS Normalization" in doc1_res.whatsapp_context.message_text
    assert "context bonus" in doc1_res.adjustment_explanation.lower()


def test_explicit_filters_precedence_over_context_ranking(tmp_path: Path):
    """Metadata filters strictly exclude documents regardless of their context relevance."""
    db_path = tmp_path / "studyrag.db"
    init_db(db_path)

    gid = get_or_create_group("CS_101", db_path=db_path)
    f1 = insert_file(gid, "h1", "f1.pdf", str(tmp_path / "f1.pdf"), ".pdf", db_path=db_path)

    # File 1 has strong context match but is from 'Student Bob'
    upsert_attachment_ref(
        group_id=gid,
        dedup_key="k1",
        status="available",
        file_id=f1,
        sender="Student Bob",
        sent_at="2026-03-25T09:00:00",
        caption="DBMS Normalization Guide",
        db_path=db_path,
    )

    with db_session(db_path) as conn:
        conn.cursor().execute(
            "INSERT INTO chunks (id, file_id, text, location) VALUES (?, ?, ?, ?)",
            ("c1", f1, "DBMS Normalization details", "Page 1"),
        )

    # Filter requires sender 'Prof. Alan'
    criteria = WhatsAppFilterCriteria(senders=["Prof. Alan"])

    res = search_ranked_documents(
        query="DBMS normalization",
        limit=10,
        db_path=db_path,
        whatsapp_filters=criteria,
        custom_reranker=lambda pairs: [4.0 for _ in pairs],
        custom_embedder=lambda texts: [[0.1] * 384 for _ in texts],
    )

    # File 1 was excluded because it didn't match sender filter, despite having matching context!
    assert len(res.documents) == 0
    assert res.excluded_documents_count == 1


def test_context_ranking_failure_fallback_preserves_content_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """If context ranking raises an exception, search gracefully falls back to pure content ranking."""
    db_path = tmp_path / "studyrag.db"
    init_db(db_path)

    gid = get_or_create_group("CS_101", db_path=db_path)
    f1 = insert_file(gid, "h1", "f1.pdf", str(tmp_path / "f1.pdf"), ".pdf", db_path=db_path)

    with db_session(db_path) as conn:
        conn.cursor().execute(
            "INSERT INTO chunks (id, file_id, text, location) VALUES (?, ?, ?, ?)",
            ("c1", f1, "DBMS Normalization details", "Page 1"),
        )

    # Monkeypatch apply_context_ranking to raise an unexpected error
    import core.context_ranking

    def buggy_apply(*args, **kwargs):
        raise RuntimeError("Simulated transient context ranking crash")

    monkeypatch.setattr(core.context_ranking, "apply_context_ranking", buggy_apply)

    res = search_ranked_documents(
        query="DBMS normalization",
        limit=10,
        db_path=db_path,
        custom_reranker=lambda pairs: [4.0 for _ in pairs],
        custom_embedder=lambda texts: [[0.1] * 384 for _ in texts],
    )

    # Search results are preserved despite context ranking failure!
    assert len(res.documents) == 1
    assert res.documents[0].file_id == f1
    assert res.is_context_ranked is False
    assert any("context ranking error" in d.lower() for d in res.diagnostics)
