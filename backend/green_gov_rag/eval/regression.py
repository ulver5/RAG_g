"""Regression gating: compare a new evaluation run against a saved baseline.

Any of the four dimensions degrading beyond tolerance fails the gate. This prevents
optimising one metric (e.g. adding an aggressive reranker cutoff that boosts
precision) from silently degrading another (e.g. hit rate or latency).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from green_gov_rag.eval.metrics import EvalMatrix

# Per-dimension tolerances. Quality metrics: allowed *drop*. Latency: allowed *rise*.
DEFAULT_TOLERANCES = {
    "retrieval_hit_rate": 0.02,  # may drop at most 2 points
    "answer_accuracy": 0.02,
    "satisfaction": 0.05,
    "latency_p99_ms": 0.20,  # may rise at most 20% (relative)
}


@dataclass
class RegressionResult:
    """Outcome of comparing current metrics to a baseline."""

    passed: bool
    failures: list[str] = field(default_factory=list)
    deltas: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "failures": self.failures, "deltas": self.deltas}


def load_baseline(path: str | Path) -> Optional[EvalMatrix]:
    """Load a saved baseline matrix, or None if the file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return EvalMatrix(**{k: v for k, v in data.items() if k in EvalMatrix().to_dict()})


def save_baseline(matrix: EvalMatrix, path: str | Path) -> None:
    """Persist a matrix as the new baseline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(matrix.to_dict(), f, indent=2)


def compare_to_baseline(
    current: EvalMatrix,
    baseline: EvalMatrix,
    tolerances: Optional[dict] = None,
) -> RegressionResult:
    """Compare ``current`` to ``baseline`` and flag regressions."""
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    failures: list[str] = []
    deltas: dict = {}

    # Quality dimensions: higher is better; fail if drop exceeds tolerance.
    for dim in ("retrieval_hit_rate", "answer_accuracy", "satisfaction"):
        cur = getattr(current, dim)
        base = getattr(baseline, dim)
        if cur is None or base is None:
            continue
        delta = cur - base
        deltas[dim] = delta
        if delta < -tol[dim]:
            failures.append(
                f"{dim} regressed by {abs(delta):.3f} (>{tol[dim]:.3f}): "
                f"{base:.3f} -> {cur:.3f}"
            )

    # Latency P99: lower is better; fail if relative rise exceeds tolerance.
    cur_lat = current.latency_p99_ms
    base_lat = baseline.latency_p99_ms
    if base_lat > 0:
        rel = (cur_lat - base_lat) / base_lat
        deltas["latency_p99_ms_rel"] = rel
        if rel > tol["latency_p99_ms"]:
            failures.append(
                f"latency_p99_ms rose {rel * 100:.1f}% (>{tol['latency_p99_ms'] * 100:.0f}%): "
                f"{base_lat:.0f}ms -> {cur_lat:.0f}ms"
            )

    return RegressionResult(passed=not failures, failures=failures, deltas=deltas)
