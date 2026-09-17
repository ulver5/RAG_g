"""Query complexity classification for tiered retrieval routing.

The full pipeline (dense + sparse + RRF + cross-encoder rerank) is accurate but
costly (~seconds). Many queries are simple lookups that a fast lexical path answers
well in tens of milliseconds. Routing trades cost/latency against accuracy:

    simple  -> fast path  (BM25 or dense only, no rerank)
    complex -> full path   (dense + sparse + RRF + rerank)

This mirrors the "分级路由" strategy: high-frequency simple queries stay cheap,
hard semantic queries get the heavy machinery. Classification is rule-based (no
extra model / latency); it can be swapped for a learned classifier later.
"""

from __future__ import annotations

import re

# Signals that a query needs deeper semantic retrieval + reranking.
_COMPLEX_KEYWORDS = (
    "compare",
    "difference",
    "differ",
    "versus",
    " vs ",
    "relationship",
    "conflict",
    "how does",
    "why",
    "explain",
    "implication",
    "interact",
    "both",
    "and also",
    "as well as",
    "step",
    "process",
    "procedure",
    "requirements for",
    "what happens if",
    "across",
    "between",
)

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

SIMPLE = "simple"
COMPLEX = "complex"


def classify_query_complexity(query: str) -> str:
    """Return ``"simple"`` or ``"complex"`` for ``query`` using cheap heuristics.

    Heuristics (any one triggers complex):
    - length: long queries tend to be multi-part / comparative
    - multiple clauses: commas / "and" / "or" joining sub-questions
    - complexity keywords: comparison, causation, process, cross-entity language
    """
    if not query or not query.strip():
        return SIMPLE

    text = f" {query.lower().strip()} "
    words = _WORD_RE.findall(text)

    # Long queries: likely multi-part.
    if len(words) >= 18:
        return COMPLEX

    # Multiple conjunctions / clauses suggest composite intent.
    conjunction_count = text.count(" and ") + text.count(" or ") + text.count(",")
    if conjunction_count >= 2:
        return COMPLEX

    # Multiple question marks -> more than one ask.
    if query.count("?") >= 2:
        return COMPLEX

    for kw in _COMPLEX_KEYWORDS:
        if kw in text:
            return COMPLEX

    return SIMPLE
