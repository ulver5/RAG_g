"""Span-level inline citations.

GreenGovRAG previously appended citation markers ``[1][2]...`` to the *end* of an
answer (documented as a known limitation). This attaches ``[n]`` markers to the
*sentences* they support, by matching each answer sentence to the source excerpt it
most overlaps with (lexical token-overlap; no extra model / latency).

This is intentionally lightweight and deterministic. It won't be as precise as an
attribution model, but it moves citations from "dumped at the end" to "aligned to
claims", and never fabricates a citation (a sentence with no meaningful overlap
gets no marker).
"""

from __future__ import annotations

import re

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common words that shouldn't drive attribution.
_STOPWORDS = frozenset(
    """the a an and or of to in for on with is are be as by from this that these those
    it its into under over their your our you we they he she which who whom whose what
    when where why how not no do does did can could should would may might must will
    shall about above below between within without such than then there here also""".split()
)

_MIN_OVERLAP = 3  # minimum shared content tokens to justify a citation
_MIN_RATIO = 0.18  # minimum (shared / sentence tokens) ratio


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS}


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [s for s in _SENTENCE_RE.split(text) if s.strip()]


def attach_span_citations(answer: str, sources: list) -> str:
    """Return ``answer`` with inline ``[n]`` markers aligned to supporting sources.

    Args:
        answer: The generated answer text.
        sources: Ordered list of source objects. Each may be a LangChain Document
            (``page_content``) or a dict/pydantic model exposing ``excerpt``/``content``.
            The 1-based index in this list is the citation number.

    Returns:
        The answer with ``[n]`` appended to sentences that overlap source ``n``.
    """
    if not answer or not sources:
        return answer

    # Pre-compute content-token sets per source.
    source_tokens: list[set[str]] = []
    for src in sources:
        if hasattr(src, "page_content"):
            text = src.page_content
        elif isinstance(src, dict):
            text = src.get("excerpt") or src.get("content") or ""
        else:
            text = getattr(src, "excerpt", "") or getattr(src, "content", "") or ""
        source_tokens.append(_content_tokens(text))

    sentences = _split_sentences(answer)
    if not sentences:
        return answer

    cited_sentences: list[str] = []
    for sentence in sentences:
        s_tokens = _content_tokens(sentence)
        if not s_tokens:
            cited_sentences.append(sentence)
            continue

        # Score each source by token overlap with this sentence.
        best_idx = -1
        best_overlap = 0
        for idx, s_src in enumerate(source_tokens):
            overlap = len(s_tokens & s_src)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx

        ratio = best_overlap / max(len(s_tokens), 1)
        if best_idx >= 0 and best_overlap >= _MIN_OVERLAP and ratio >= _MIN_RATIO:
            marker = f" [{best_idx + 1}]"
            # Insert the marker before a trailing terminal punctuation if present.
            stripped = sentence.rstrip()
            if stripped and stripped[-1] in ".!?":
                cited_sentences.append(f"{stripped[:-1]}{marker}{stripped[-1]}")
            else:
                cited_sentences.append(f"{stripped}{marker}")
        else:
            cited_sentences.append(sentence)

    return " ".join(cited_sentences)
