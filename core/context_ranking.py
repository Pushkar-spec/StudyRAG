"""WhatsApp context-aware ranking and conservative score combination for StudyRAG.

Evaluates query relevance against WhatsApp sharing messages and conversational context,
and conservatively combines context scores with document content relevance scores to
resolve close results without overpowering content-level cross-encoder rankings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
import re
from typing import Any

from core.whatsapp_context import DocumentWhatsAppContext, WhatsAppMessageContext

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CONTEXT_WEIGHT",
    "WhatsAppContextScore",
    "ContextRankingAdjustment",
    "extract_query_tokens",
    "score_text_overlap",
    "score_whatsapp_context",
    "combine_relevance_scores",
    "apply_context_ranking",
]

DEFAULT_CONTEXT_WEIGHT: float = 0.08

# Standard lightweight English stopwords to prevent uninformative word matches
# (e.g. 'notes on dbms' should match 'notes' and 'dbms', not 'on')
_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
})


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass
class WhatsAppContextScore:
    """Component scores and explanation for a document's WhatsApp conversational context."""

    direct_score: float  # Bounded [0.0, 1.0] score from direct sharing message / caption
    surrounding_score: float  # Bounded [0.0, 1.0] score from surrounding conversation window
    combined_score: float  # Bounded [0.0, 1.0] weighted composite context relevance score
    matched_terms: list[str] = field(default_factory=list)
    has_context: bool = False
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to standard dictionary."""
        return asdict(self)


@dataclass
class ContextRankingAdjustment:
    """Full breakdown of ranking adjustments and score contributions for a document."""

    file_id: int
    filename: str
    original_score: float
    original_rank: int
    context_score: WhatsAppContextScore
    boost: float
    final_score: float
    final_rank: int
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to standard dictionary."""
        d = asdict(self)
        d["context_score"] = self.context_score.to_dict()
        return d


# ---------------------------------------------------------------------------
# Tokenization and Overlap Scoring Helpers
# ---------------------------------------------------------------------------

def extract_query_tokens(query: str) -> list[str]:
    """Tokenize query string into normalized alphanumeric terms, filtering English stopwords.

    Preserves multi-word content terms while removing punctuation and high-frequency noise.
    Falls back to all alphanumeric terms if stopwords remove all tokens.
    """
    clean_query = query.strip().lower() if query else ""
    if not clean_query:
        return []

    raw_tokens = re.findall(r"\b\w+\b", clean_query)
    if not raw_tokens:
        return []

    # Filter out stopwords and single characters
    content_tokens = [t for t in raw_tokens if len(t) > 1 and t not in _STOPWORDS]
    if not content_tokens:
        # Fallback to alphanumeric tokens longer than 1 character
        content_tokens = [t for t in raw_tokens if len(t) > 1]
    if not content_tokens:
        content_tokens = raw_tokens

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for token in content_tokens:
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result


def score_text_overlap(
    query_tokens: list[str],
    query_phrase: str,
    text: str | None,
) -> tuple[float, list[str]]:
    """Calculate bounded lexical relevance [0.0, 1.0] and matched terms between query and text.

    Scoring logic:
    1. Token overlap coverage: Fraction of query terms present in text tokens.
    2. Exact phrase bonus (+0.15): Awarded if the entire query phrase occurs as an exact substring.
    3. Final score = min(1.0, 0.85 * coverage + phrase_bonus).
    """
    if not text or not text.strip() or not query_tokens:
        return 0.0, []

    text_lower = text.lower()
    text_tokens = set(re.findall(r"\b\w+\b", text_lower))

    matched = [t for t in query_tokens if t in text_tokens]
    if not matched:
        return 0.0, []

    coverage = len(matched) / len(query_tokens)

    # Check for exact multi-word phrase match
    phrase_bonus = 0.0
    clean_phrase = query_phrase.strip().lower() if query_phrase else ""
    if len(query_tokens) >= 2 and clean_phrase and clean_phrase in text_lower:
        phrase_bonus = 0.15

    raw_score = 0.85 * coverage + phrase_bonus
    bounded_score = round(min(1.0, max(0.0, raw_score)), 4)
    return bounded_score, matched


# ---------------------------------------------------------------------------
# Context Relevance Scoring
# ---------------------------------------------------------------------------

def score_whatsapp_context(
    query: str,
    document_context: DocumentWhatsAppContext | None,
) -> WhatsAppContextScore:
    """Evaluate query relevance against a document's WhatsApp conversational context.

    Evaluates:
    - Direct sharing message / caption and original filename (primary context).
    - Surrounding conversation window (secondary context).

    Weighting strategy:
    - If both direct caption and surrounding conversation exist:
      `combined = 0.75 * direct_score + 0.25 * surrounding_score`
    - If only direct caption exists:
      `combined = direct_score`
    - If only surrounding conversation exists:
      `combined = 0.70 * surrounding_score` (discounted for indirect conversation)
    - Missing or unlinked context produces a neutral score (0.0), never an automatic penalty.

    Args:
        query: User search query or syllabus topic.
        document_context: WhatsApp context retrieved for the candidate document.

    Returns:
        WhatsAppContextScore instance with component scores, matched terms, and explanation.
    """
    if document_context is None or not document_context.has_linked_message:
        return WhatsAppContextScore(
            direct_score=0.0,
            surrounding_score=0.0,
            combined_score=0.0,
            matched_terms=[],
            has_context=False,
            explanation="Ranked by document content relevance alone (no linked WhatsApp context).",
        )

    clean_query = query.strip() if query else ""
    if not clean_query:
        return WhatsAppContextScore(
            direct_score=0.0,
            surrounding_score=0.0,
            combined_score=0.0,
            matched_terms=[],
            has_context=True,
            explanation="Empty query provided; zero context relevance.",
        )

    query_tokens = extract_query_tokens(clean_query)
    clean_phrase = " ".join(re.findall(r"\b\w+\b", clean_query.lower()))

    # 1. Direct Sharing Message & Attachment Metadata
    primary_msg = document_context.primary_message
    direct_caption = (
        document_context.message_text
        or (primary_msg.message_text if primary_msg else None)
        or ""
    )
    orig_filename = (primary_msg.original_filename if primary_msg else None) or ""

    direct_source = f"{direct_caption} {orig_filename}".strip()
    direct_score, direct_matched = score_text_overlap(query_tokens, clean_phrase, direct_source)

    # 2. Surrounding Conversational Context (5-min window)
    surr_context = (
        document_context.surrounding_context
        or (primary_msg.surrounding_context if primary_msg else None)
        or ""
    )
    surr_score, surr_matched = score_text_overlap(query_tokens, clean_phrase, surr_context)

    # 3. Composite Context Score Calculation
    if direct_score > 0.0 and surr_score > 0.0:
        combined = 0.75 * direct_score + 0.25 * surr_score
    elif direct_score > 0.0:
        combined = direct_score
    elif surr_score > 0.0:
        combined = 0.70 * surr_score
    else:
        combined = 0.0

    combined_score = round(min(1.0, max(0.0, combined)), 4)

    # Deduplicate matched terms preserving encounter order
    seen_terms: set[str] = set()
    all_matched: list[str] = []
    for term in direct_matched + surr_matched:
        if term not in seen_terms:
            seen_terms.add(term)
            all_matched.append(term)

    # Construct factual, transparent explanation
    if combined_score > 0.0:
        matched_str = f"[{', '.join(all_matched)}]"
        if direct_score > 0.0 and surr_score > 0.0:
            explanation = f"WhatsApp sharing message and surrounding discussion matched query terms {matched_str}."
        elif direct_score > 0.0:
            explanation = f"WhatsApp sharing message matched query terms {matched_str}."
        else:
            explanation = f"WhatsApp surrounding conversation matched query terms {matched_str}."
    else:
        explanation = "WhatsApp context linked, but message and conversation did not match query terms."

    return WhatsAppContextScore(
        direct_score=direct_score,
        surrounding_score=surr_score,
        combined_score=combined_score,
        matched_terms=all_matched,
        has_context=True,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Score Combination & Ranking Adjustment
# ---------------------------------------------------------------------------

def combine_relevance_scores(
    content_score: float,
    context_score: float,
    score_range: float = 1.0,
    weight: float = DEFAULT_CONTEXT_WEIGHT,
) -> tuple[float, float]:
    """Combine document content relevance and WhatsApp context relevance conservatively.

    Formula:
        effective_range = max(score_range, 1.0)
        boost = weight * effective_range * context_score
        final_score = content_score + boost

    Properties & Guarantees:
    1. Content Dominance: Context weight is conservative (default 0.08, capped <= 0.50).
       A document with weak content relevance cannot overcome a strong content match solely
       by having matching terms in its WhatsApp caption.
    2. Missing Context Neutrality: If context_score == 0.0, boost is strictly 0.0 and
       final_score == content_score. Missing context never incurs a penalty.
    3. Scale Invariance: Multiplying by effective_range bounds the maximum possible boost
       to at most `weight * dynamic_range`, preventing cross-encoder logits or hybrid scores
       from being overwhelmed regardless of their absolute scale.
    4. Deterministic and Monotonic: Higher content and context relevance monotonically
       yield higher final scores.

    Args:
        content_score: Raw document score from cross-encoder or hybrid fallback.
        context_score: Normalized WhatsApp context score in [0.0, 1.0].
        score_range: Difference between maximum and minimum content scores in candidate pool.
        weight: Bounded context influence weight (0.0 to 0.50).

    Returns:
        Tuple of (final_score, boost).
    """
    safe_weight = max(0.0, min(0.50, float(weight)))
    safe_context_score = max(0.0, min(1.0, float(context_score)))
    effective_range = max(1.0, float(score_range))

    boost = round(safe_weight * effective_range * safe_context_score, 4)
    final_score = round(content_score + boost, 4)
    return final_score, boost


def apply_context_ranking(
    documents: list[Any],
    query: str,
    context_map: dict[int, DocumentWhatsAppContext] | None = None,
    weight: float = DEFAULT_CONTEXT_WEIGHT,
    disable: bool = False,
) -> tuple[list[Any], list[ContextRankingAdjustment]]:
    """Apply WhatsApp context relevance scoring and conservative reranking to candidate documents.

    Pipeline:
    1. Records original content scores and ranks.
    2. Computes the dynamic content score range across candidates.
    3. Evaluates WhatsApp context relevance for each candidate document.
    4. Combines content and context scores using the bounded additive boost formula.
    5. Re-sorts documents deterministically:
       `(-final_score, -original_content_score, -chunk_count, file_id)`
    6. Re-assigns sequential ranks (1, 2, ...) and generates factual adjustment explanations.

    Args:
        documents: List of RankedDocument instances.
        query: User search query.
        context_map: Optional pre-retrieved DocumentWhatsAppContext lookup by file_id.
        weight: Bounded context weight (default 0.08).
        disable: If True, bypass context ranking adjustments and preserve pure content order.

    Returns:
        Tuple of (adjusted_documents, adjustments_list).
    """
    if not documents:
        return [], []

    adjustments: list[ContextRankingAdjustment] = []

    # Record baseline content scores and content ranks if not already recorded
    for idx, doc in enumerate(documents, start=1):
        if getattr(doc, "content_score", None) is None:
            doc.content_score = float(doc.document_score)
        if getattr(doc, "content_rank", None) is None:
            doc.content_rank = int(doc.rank or idx)

    # If context ranking is disabled, annotate documents and return unchanged
    if disable or weight <= 0.0:
        for doc in documents:
            doc.adjustment_explanation = "Ranked by document content relevance alone (context ranking disabled)."
            adj = ContextRankingAdjustment(
                file_id=doc.file_id,
                filename=doc.filename,
                original_score=doc.content_score,
                original_rank=doc.content_rank,
                context_score=WhatsAppContextScore(
                    direct_score=0.0,
                    surrounding_score=0.0,
                    combined_score=0.0,
                    has_context=False,
                    explanation="Context ranking disabled.",
                ),
                boost=0.0,
                final_score=doc.content_score,
                final_rank=doc.content_rank,
                explanation=doc.adjustment_explanation,
            )
            adjustments.append(adj)
        return documents, adjustments

    # Compute dynamic content score range across candidate documents
    content_scores = [float(d.content_score) for d in documents]
    c_max = max(content_scores)
    c_min = min(content_scores)
    score_range = c_max - c_min

    # Calculate context scores and boosted final scores
    temp_records: list[dict[str, Any]] = []

    for doc in documents:
        # Retrieve context from doc property or lookup map
        ctx: DocumentWhatsAppContext | None = getattr(doc, "whatsapp_context", None)
        if ctx is None and context_map:
            ctx = context_map.get(doc.file_id)

        ctx_score = score_whatsapp_context(query=query, document_context=ctx)
        final_score, boost = combine_relevance_scores(
            content_score=doc.content_score,
            context_score=ctx_score.combined_score,
            score_range=score_range,
            weight=weight,
        )

        temp_records.append({
            "doc": doc,
            "ctx_score": ctx_score,
            "boost": boost,
            "final_score": final_score,
        })

    # Sort documents deterministically:
    # 1. Highest final_score
    # 2. Highest original content_score (content priority tie-breaker)
    # 3. Highest chunk_count (more supporting evidence)
    # 4. Lowest file_id (database determinism)
    temp_records.sort(
        key=lambda r: (
            -r["final_score"],
            -r["doc"].content_score,
            -getattr(r["doc"], "chunk_count", 0),
            r["doc"].file_id,
        )
    )

    # Re-assign sequential ranks and attach diagnostic explanations
    reordered_docs: list[Any] = []

    for new_rank, item in enumerate(temp_records, start=1):
        doc = item["doc"]
        ctx_score: WhatsAppContextScore = item["ctx_score"]
        boost: float = item["boost"]
        final_score: float = item["final_score"]
        orig_rank: int = doc.content_rank

        # Construct concise, factual adjustment explanation
        matched_str = f"[{', '.join(ctx_score.matched_terms)}]" if ctx_score.matched_terms else ""
        if new_rank < orig_rank:
            explanation = (
                f"Rank improved from #{orig_rank} to #{new_rank} "
                f"(+{boost:.3f} context bonus from WhatsApp context matching {matched_str})."
            )
        elif new_rank > orig_rank:
            explanation = (
                f"Rank shifted from #{orig_rank} to #{new_rank} because other candidate documents "
                f"received higher WhatsApp context relevance."
            )
        elif boost > 0.0001:
            explanation = (
                f"Rank maintained at #{new_rank} (+{boost:.3f} context bonus from WhatsApp context matching {matched_str})."
            )
        elif not ctx_score.has_context:
            explanation = "Ranked by document content relevance alone (no linked WhatsApp sharing message)."
        else:
            explanation = "Ranked by document content relevance alone (linked WhatsApp context did not match query terms)."

        # Update document attributes
        doc.rank = new_rank
        doc.document_score = final_score
        doc.context_score = ctx_score
        doc.adjustment_explanation = explanation

        reordered_docs.append(doc)

        adjustments.append(
            ContextRankingAdjustment(
                file_id=doc.file_id,
                filename=doc.filename,
                original_score=doc.content_score,
                original_rank=orig_rank,
                context_score=ctx_score,
                boost=boost,
                final_score=final_score,
                final_rank=new_rank,
                explanation=explanation,
            )
        )

    return reordered_docs, adjustments
