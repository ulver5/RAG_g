"""Pre-generation confidence gating (hallucination control).

GreenGovRAG previously controlled hallucination only *after* generation (trust
score, citation verification). This adds a *pre-generation* gate driven by
retrieval confidence, so we decide whether to answer at all before spending an
LLM call and before risking an unsupported answer:

    top retrieval score >= high threshold (0.85)  -> ANSWER  (cite sources)
    low <= score < high                            -> CAVEAT  (answer + "may be incomplete")
    score < low threshold (0.60)                   -> CLARIFY (ask a question, don't answer)

The score is the reranker score when available (0-1, most reliable), otherwise the
normalised fusion/similarity score attached by the retrieval pipeline. Low-confidence
cases feed the bad-case loop for weekly review.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from green_gov_rag.config import settings


class AnswerType(str, Enum):
    """How the response should be treated by the API / UI."""

    ANSWERED = "answered"  # high confidence, cite sources
    PARTIAL = "partial"  # medium confidence, answer with caveat
    CLARIFICATION = "clarification"  # low confidence, ask back instead of answering


@dataclass
class ConfidenceDecision:
    """Outcome of the confidence gate."""

    score: float
    level: str  # "high" | "medium" | "low"
    answer_type: AnswerType
    should_generate: bool  # False => return a clarification, skip the LLM
    caveat: Optional[str] = None
    clarification: Optional[str] = None


def _top_score(documents: list) -> float:
    """Best available 0-1 relevance score across retrieved documents."""
    best = 0.0
    for doc in documents:
        meta = getattr(doc, "metadata", {}) or {}
        # Prefer rerank score, then fused/normalised score, then any raw score.
        for key in ("rerank_score", "score", "relevance_score"):
            val = meta.get(key)
            if isinstance(val, (int, float)):
                best = max(best, float(val))
                break
    return best


def _build_clarification(query: str, documents: list) -> str:
    """Compose a helpful clarifying question for low-confidence queries."""
    if not documents:
        return (
            "I couldn't find regulations that clearly match your question. "
            "Could you add more detail — for example the jurisdiction (federal / a "
            "specific state) or Local Government Area, and the activity or topic "
            "you're asking about?"
        )
    # Some weak matches exist: surface topics and ask the user to narrow down.
    topics = []
    for doc in documents[:3]:
        meta = getattr(doc, "metadata", {}) or {}
        t = meta.get("topic") or meta.get("category")
        if t and t not in topics:
            topics.append(str(t))
    topic_hint = f" (nearby topics: {', '.join(topics)})" if topics else ""
    return (
        "I found some possibly-related material but nothing confidently on point"
        f"{topic_hint}. Could you clarify the jurisdiction or region, and be more "
        "specific about the activity or requirement you need?"
    )


def assess_confidence(query: str, documents: list) -> ConfidenceDecision:
    """Assess retrieval confidence and decide the answer strategy.

    When gating is disabled, always answers (no caveat / no clarify) to preserve
    legacy behaviour.
    """
    score = _top_score(documents)

    if not settings.enable_confidence_gating:
        return ConfidenceDecision(
            score=score,
            level="high",
            answer_type=AnswerType.ANSWERED,
            should_generate=True,
        )

    high = settings.confidence_high_threshold
    low = settings.confidence_low_threshold

    if not documents or score < low:
        return ConfidenceDecision(
            score=score,
            level="low",
            answer_type=AnswerType.CLARIFICATION,
            should_generate=False,
            clarification=_build_clarification(query, documents),
        )

    if score < high:
        return ConfidenceDecision(
            score=score,
            level="medium",
            answer_type=AnswerType.PARTIAL,
            should_generate=True,
            caveat=(
                "⚠️ This answer may be incomplete or not fully on point — the "
                "retrieved sources are only a partial match. Please verify against "
                "the cited documents."
            ),
        )

    return ConfidenceDecision(
        score=score,
        level="high",
        answer_type=AnswerType.ANSWERED,
        should_generate=True,
    )
