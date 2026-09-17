"""RAG evaluation harness: the four-dimension quality matrix + regression gating.

GreenGovRAG had no retrieval/answer quality baseline. This package implements the
metric-driven loop:

    1. retrieval hit rate  - did we retrieve a chunk that contains the answer?
    2. answer accuracy      - is the generated answer correct (keyword / LLM judge)?
    3. latency P99          - first-response latency at the tail
    4. satisfaction         - thumbs up/down (feedback ratings)

Plus regression gating: compare a new run against a saved baseline and fail if any
dimension degrades beyond tolerance, so a single-metric optimisation can't silently
degrade the whole experience.
"""

from green_gov_rag.eval.dataset import EvalItem, item_is_hit, load_dataset
from green_gov_rag.eval.metrics import (
    EvalMatrix,
    compute_answer_accuracy,
    compute_latency_metrics,
    compute_retrieval_metrics,
    compute_satisfaction,
    percentile,
)
from green_gov_rag.eval.regression import RegressionResult, compare_to_baseline

__all__ = [
    "EvalItem",
    "load_dataset",
    "item_is_hit",
    "EvalMatrix",
    "compute_retrieval_metrics",
    "compute_answer_accuracy",
    "compute_latency_metrics",
    "compute_satisfaction",
    "percentile",
    "RegressionResult",
    "compare_to_baseline",
]
