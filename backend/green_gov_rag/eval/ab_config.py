"""A/B config loading + application for offline evaluation.

A config file (YAML or JSON) describes a set of ``settings`` field overrides that
define one system configuration ("A" or "B"). The A/B tool runs the SAME eval
dataset under each config and diffs the metrics.

File shape (flat)::

    name: hybrid_rerank            # optional label for the report
    description: BM25 + RRF + rerank
    k: 5                           # optional retrieval depth (method arg, not a setting)
    enable_hybrid_retrieval: true  # any real Settings field below
    enable_bm25: true
    enable_reranker: true
    rrf_k: 60

Everything except the reserved keys (``name``/``description``/``k``) is treated as
a ``settings`` field override and validated against the real ``Settings`` model.

Design notes
------------
- Overrides are applied by **mutating the settings singleton's fields** — safe
  because the retrieval/confidence paths read ``settings.*`` live per call. For
  full isolation across the two configs, the A/B tool runs each in its own
  subprocess (see ``scripts/ab_compare.py``); this module only applies one config
  to the current process.
- Caches are **force-disabled** on every applied config, otherwise config B can
  return config A's cached answers and corrupt the comparison.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Keys in a config file that are NOT settings overrides.
RESERVED_KEYS = {"name", "description", "k"}

# Caches that must be off during A/B, so B never serves A's cached results.
FORCED_OFF = {"enable_cache": False, "enable_semantic_cache": False}

# Knobs that meaningfully change retrieval/generation behaviour and take effect
# via live reads inside the (subprocess) run. Overriding a real Settings field
# outside this set is allowed but warned about (e.g. elasticsearch_url).
KNOWN_AB_KNOBS = {
    "enable_hybrid_retrieval",
    "enable_bm25",
    "bm25_backend",
    "recall_k",
    "rrf_k",
    "enable_reranker",
    "reranker_model",
    "rerank_top_n",
    "enable_tiered_routing",
    "enable_confidence_gating",
    "confidence_high_threshold",
    "confidence_low_threshold",
    "enable_span_citations",
    "llm_provider",
    "llm_model",
}


@dataclass
class ABConfig:
    """One parsed A/B configuration."""

    name: str
    description: str = ""
    k: Optional[int] = None  # retrieval depth (top-k), a method arg not a setting
    overrides: dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "k": self.k,
            "overrides": self.overrides,
            "source_path": self.source_path,
        }


def load_config(path: str | Path) -> ABConfig:
    """Load and parse an A/B config file (``.yaml``/``.yml``/``.json``)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml  # PyYAML is a declared dependency

        raw = yaml.safe_load(text) or {}
    elif path.suffix.lower() == ".json":
        raw = json.loads(text) if text.strip() else {}
    else:
        msg = f"Unsupported config extension: {path.suffix} (use .yaml/.yml/.json)"
        raise ValueError(msg)

    if not isinstance(raw, dict):
        msg = f"Config {path} must be a mapping of keys to values, got {type(raw).__name__}"
        raise ValueError(msg)

    name = str(raw.get("name") or path.stem)
    description = str(raw.get("description") or "")
    k = raw.get("k")
    if k is not None:
        k = int(k)

    overrides = {key: val for key, val in raw.items() if key not in RESERVED_KEYS}
    return ABConfig(
        name=name, description=description, k=k, overrides=overrides, source_path=str(path)
    )


def _known_fields(settings: Any) -> set[str]:
    """Return the set of valid field names on a Settings-like object."""
    model_fields = getattr(type(settings), "model_fields", None)
    if model_fields:
        return set(model_fields)
    # Fallback for plain objects (e.g. the offline self-test stub).
    return set(vars(settings))


def apply_overrides(settings: Any, overrides: dict[str, Any]) -> dict[str, Any]:
    """Mutate ``settings`` with ``overrides`` (+ forced cache-off).

    Validates every key against the real Settings fields; an unknown field is a
    hard error (typo protection). Fields outside :data:`KNOWN_AB_KNOBS` are applied
    but warned about. Returns the effective override dict actually applied (for the
    report), including the forced cache flags.
    """
    valid = _known_fields(settings)
    effective: dict[str, Any] = {}

    for key, value in overrides.items():
        if key not in valid:
            msg = f"Unknown settings field in config: {key!r} (not a Settings attribute)"
            raise ValueError(msg)
        if key not in KNOWN_AB_KNOBS:
            logger.warning("Override %r is a valid Settings field but not a typical A/B knob", key)
        setattr(settings, key, value)
        effective[key] = value

    # Force caches off last, so a config can't accidentally re-enable them.
    for key, value in FORCED_OFF.items():
        if key in valid:
            setattr(settings, key, value)
            effective[key] = value

    return effective


# ---------------------------------------------------------------------------
# Offline self-test (no infra / API key required)
# ---------------------------------------------------------------------------
def _selftest() -> int:
    import tempfile
    import types

    ok = True

    def _check(label: str, cond: bool) -> None:
        nonlocal ok
        ok &= cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    print("== load_config (yaml) ==")
    yaml_text = (
        "name: hybrid_rerank\n"
        "description: BM25 + RRF + rerank\n"
        "k: 5\n"
        "enable_hybrid_retrieval: true\n"
        "enable_bm25: true\n"
        "enable_reranker: true\n"
        "rrf_k: 60\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
        fh.write(yaml_text)
        yaml_path = fh.name
    cfg = load_config(yaml_path)
    _check("name parsed", cfg.name == "hybrid_rerank")
    _check("k parsed as int", cfg.k == 5 and isinstance(cfg.k, int))
    _check("reserved keys excluded from overrides", "name" not in cfg.overrides and "k" not in cfg.overrides)
    _check("settings override captured", cfg.overrides.get("rrf_k") == 60)

    print("\n== apply_overrides ==")
    fake = types.SimpleNamespace(
        enable_hybrid_retrieval=False, enable_bm25=False, enable_reranker=False,
        rrf_k=0, enable_cache=True, enable_semantic_cache=True, unrelated="x",
    )
    effective = apply_overrides(fake, cfg.overrides)
    _check("hybrid applied", fake.enable_hybrid_retrieval is True)
    _check("rrf_k applied", fake.rrf_k == 60)
    _check("cache forced off", fake.enable_cache is False and fake.enable_semantic_cache is False)
    _check("forced flags in effective", effective.get("enable_cache") is False)

    print("\n== unknown field rejected ==")
    try:
        apply_overrides(fake, {"not_a_real_field": 1})
        _check("raised ValueError", False)
    except ValueError:
        _check("raised ValueError", True)

    print(f"\nSELF-TEST {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print("Usage: python green_gov_rag/eval/ab_config.py --selftest")
