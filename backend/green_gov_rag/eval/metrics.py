"""Metric computations for the four-dimension evaluation matrix."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def percentile(values: list[float], p: float) -> float:
    """Return the ``p``-th percentile (0-100) of ``values`` (nearest-rank)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


@dataclass
class EvalMatrix:
    """The four-dimension quality snapshot for one run."""

    retrieval_hit_rate: float = 0.0
    retrieval_mrr: float = 0.0
    answer_accuracy: Optional[float] = None
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_mean_ms: float = 0.0
    satisfaction: Optional[float] = None
    satisfaction_sample_size: int = 0
    num_queries: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def compute_retrieval_metrics(hits: list[bool], ranks: list[Optional[int]]) -> dict:
    """Compute hit rate and MRR.

    Args:
        hits: per-query boolean, True if a relevant doc was retrieved in top-k.
        ranks: per-query 1-based rank of the first relevant doc (None if miss).
    """
    n = len(hits)
    if n == 0:
        return {"hit_rate": 0.0, "mrr": 0.0}
    hit_rate = sum(1 for h in hits if h) / n
    mrr = sum((1.0 / r) for r in ranks if r) / n
    return {"hit_rate": hit_rate, "mrr": mrr}


def compute_latency_metrics(latencies_ms: list[float]) -> dict:
    """Compute latency distribution metrics (P50/P95/P99/mean)."""
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {
        "p50": percentile(latencies_ms, 50),
        "p95": percentile(latencies_ms, 95),
        "p99": percentile(latencies_ms, 99),
        "mean": sum(latencies_ms) / len(latencies_ms),
    }


def compute_answer_accuracy(judgements: list[float]) -> Optional[float]:
    """Average of per-answer correctness scores in [0,1]; None if no judgements."""
    if not judgements:
        return None
    return sum(judgements) / len(judgements)


def keyword_answer_score(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords present in the answer (cheap accuracy proxy)."""
    if not expected_keywords:
        return 0.0
    answer_l = (answer or "").lower()
    present = sum(1 for k in expected_keywords if k.lower() in answer_l)
    return present / len(expected_keywords)


def compute_satisfaction(window_days: int = 30) -> tuple[Optional[float], int]:
    """Compute satisfaction from stored feedback ratings over the last window.

    Returns ``(satisfaction, sample_size)`` where satisfaction is the mean rating
    normalised to 0-1 (rating 1-5 -> 0-1). Returns ``(None, 0)`` if unavailable.
    """
    try:
        from datetime import datetime, timedelta, timezone

        from sqlmodel import Session, select

        from green_gov_rag.models import QueryHistory
        from green_gov_rag.models.base import engine

        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        with Session(engine) as session:
            rows = session.exec(
                select(QueryHistory.feedback_rating).where(
                    QueryHistory.feedback_rating.is_not(None),
                    QueryHistory.created_at >= cutoff,
                )
            ).all()
        ratings = [r for r in rows if r is not None]
        if not ratings:
            return None, 0
        mean_rating = sum(ratings) / len(ratings)
        return (mean_rating - 1) / 4.0, len(ratings)  # 1..5 -> 0..1
    except Exception as exc:  # noqa: BLE001
        logger.warning("Satisfaction computation unavailable: %s", exc)
        return None, 0
