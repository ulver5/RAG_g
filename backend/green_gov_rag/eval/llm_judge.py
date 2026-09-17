"""LLM-as-judge scorers for answer quality — replaces the weak keyword proxy.

Background
----------
The four-dimension matrix scores *answer accuracy* with ``keyword_answer_score``
(fraction of expected keywords present in the answer). That proxy is weak: it does
not penalise hallucination and never checks whether citations are correct. This
module adds three LLM judges that score an answer on 0..1:

- ``judge_answer_correctness``  — does the answer correctly answer the query,
  agreeing with the reference answer? (substitute for keyword accuracy)
- ``judge_faithfulness``        — is every claim in the answer supported by the
  retrieved context? (catches hallucination / ungrounded claims)
- ``judge_citation_accuracy``   — do the cited sources actually contain what the
  answer attributes to them? (catches mis-citation)

Design (mirrors ``CitationVerificationService._verify_relevance``)
------------------------------------------------------------------
- Uses the project's multi-provider LLM factory (``get_llm``): OpenAI / Azure /
  Bedrock / Anthropic, whatever ``settings`` selects.
- ``temperature=0`` for determinism.
- Async ``ainvoke``; a sync ``run_judges`` convenience wrapper is provided.
- Forces **strict JSON** output and parses it robustly (``_parse_judge_json``),
  with a safe fallback so one bad response never crashes an eval run.
- Cost control: judges are meant to run on a *sample* or on *bad cases*, not on
  every request. The caller decides sampling.
- Testability: every judge accepts an injected ``llm`` object (anything with an
  ``ainvoke(prompt) -> obj.content`` method), so it can be exercised offline with
  a fake LLM. Run ``python -m green_gov_rag.eval.llm_judge --selftest``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class JudgeScore:
    """One judge's verdict on a single answer."""

    metric: str
    score: float  # 0..1
    reasoning: str = ""
    ok: bool = True  # False if the LLM call/parse failed and we fell back
    raw: str = ""  # raw model text, kept for debugging / audit

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "score": self.score,
            "reasoning": self.reasoning,
            "ok": self.ok,
        }


class SupportsAInvoke(Protocol):
    """Minimal interface we need from an LLM (LangChain chat models satisfy it)."""

    async def ainvoke(self, prompt: str) -> Any: ...


# ---------------------------------------------------------------------------
# Parsing (pure function — unit-testable without any LLM)
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)


def _parse_judge_json(text: str) -> tuple[Optional[float], str]:
    """Extract ``(score, reasoning)`` from a model's JSON reply.

    Tolerates code fences and surrounding prose by grabbing the first ``{..}``
    block. Returns ``(None, reason)`` if no usable score is found; the score is
    clamped to [0, 1].
    """
    if not text:
        return None, "empty response"

    cleaned = _FENCE_RE.sub("", text).strip()

    # Grab the first balanced-looking {...} block.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    candidate = cleaned[start : end + 1] if start != -1 and end > start else cleaned

    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        # Last resort: a bare number somewhere in the text.
        m = re.search(r"-?\d*\.?\d+", cleaned)
        if m:
            try:
                return _clamp(float(m.group())), "parsed bare number (no JSON)"
            except ValueError:
                pass
        return None, f"unparseable response: {cleaned[:80]!r}"

    # A bare JSON number (e.g. "0.42") is a valid score on its own.
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        return _clamp(float(data)), "bare number"

    if not isinstance(data, dict) or "score" not in data:
        return None, f"json missing 'score': {candidate[:80]!r}"

    try:
        score = _clamp(float(data["score"]))
    except (TypeError, ValueError):
        return None, f"non-numeric score: {data.get('score')!r}"

    reasoning = str(data.get("reasoning", "")).strip()
    return score, reasoning


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
_JSON_INSTR = (
    'Respond with ONLY a JSON object: {"score": <float 0..1>, '
    '"reasoning": "<one short sentence>"}. No prose, no code fence.'
)


def _prompt_correctness(query: str, answer: str, reference_answer: str) -> str:
    return f"""You are grading a compliance assistant's answer against a reference answer.

QUERY:
{query}

CANDIDATE ANSWER:
{answer}

REFERENCE ANSWER (ground truth):
{reference_answer}

Score how correct and complete the candidate is versus the reference:
- 1.0 = fully correct, all key facts present, no contradictions
- 0.5 = partially correct or missing important facts
- 0.0 = wrong, contradicts the reference, or non-answer

{_JSON_INSTR}"""


def _prompt_faithfulness(answer: str, contexts: list[str]) -> str:
    joined = "\n\n".join(f"[CTX {i + 1}] {c}" for i, c in enumerate(contexts)) or "(no context)"
    return f"""You are checking an answer for hallucination against retrieved context.

RETRIEVED CONTEXT:
{joined}

ANSWER:
{answer}

Break the answer into factual claims. Score the fraction that are SUPPORTED by
the context above (not by outside knowledge):
- 1.0 = every claim is supported by the context
- 0.0 = no claim is supported / the answer invents facts
General/hedging statements ("consult a professional") are neutral, ignore them.

{_JSON_INSTR}"""


def _prompt_citation_accuracy(answer: str, sources: list[dict]) -> str:
    lines = []
    for i, s in enumerate(sources):
        title = s.get("title", "Unknown")
        excerpt = s.get("excerpt") or s.get("content") or ""
        lines.append(f"[SRC {i + 1}] {title}\n{excerpt}")
    joined = "\n\n".join(lines) or "(no sources)"
    return f"""You are verifying that an answer's claims are actually backed by its cited sources.

CITED SOURCES:
{joined}

ANSWER:
{answer}

Score how accurately the answer's statements are attributable to the cited sources:
- 1.0 = every substantive statement is genuinely supported by a cited source
- 0.5 = some statements are supported, others are not traceable to any source
- 0.0 = the sources do not support what the answer claims (mis-citation)

{_JSON_INSTR}"""


# ---------------------------------------------------------------------------
# LLM plumbing
# ---------------------------------------------------------------------------
def _default_llm() -> SupportsAInvoke:
    """Build the configured judge LLM lazily (temperature 0 for determinism)."""
    from green_gov_rag.rag.llm_factory import get_llm

    return get_llm(temperature=0, max_tokens=200)


async def _run_judge(prompt: str, metric: str, llm: Optional[SupportsAInvoke]) -> JudgeScore:
    """Invoke the LLM, parse JSON, and wrap in a JudgeScore with safe fallback."""
    engine = llm or _default_llm()
    try:
        response = await engine.ainvoke(prompt)
        raw = getattr(response, "content", None)
        raw = raw if isinstance(raw, str) else str(response)
    except Exception as exc:  # noqa: BLE001 — one bad call must not kill the run
        logger.warning("Judge %s LLM call failed: %s", metric, exc)
        return JudgeScore(metric=metric, score=0.5, reasoning=f"llm error: {exc}", ok=False)

    score, reasoning = _parse_judge_json(raw)
    if score is None:
        logger.warning("Judge %s parse failed: %s", metric, reasoning)
        return JudgeScore(metric=metric, score=0.5, reasoning=reasoning, ok=False, raw=raw)
    return JudgeScore(metric=metric, score=score, reasoning=reasoning, ok=True, raw=raw)


# ---------------------------------------------------------------------------
# The three judges
# ---------------------------------------------------------------------------
async def judge_answer_correctness(
    query: str, answer: str, reference_answer: str, *, llm: Optional[SupportsAInvoke] = None
) -> JudgeScore:
    """Score answer correctness vs a reference answer (0..1)."""
    if not reference_answer:
        return JudgeScore("answer_correctness", 0.5, "no reference answer", ok=False)
    return await _run_judge(
        _prompt_correctness(query, answer, reference_answer), "answer_correctness", llm
    )


async def judge_faithfulness(
    answer: str, contexts: list[str], *, llm: Optional[SupportsAInvoke] = None
) -> JudgeScore:
    """Score the fraction of answer claims grounded in the retrieved context (0..1)."""
    return await _run_judge(_prompt_faithfulness(answer, contexts), "faithfulness", llm)


async def judge_citation_accuracy(
    answer: str, sources: list[dict], *, llm: Optional[SupportsAInvoke] = None
) -> JudgeScore:
    """Score whether the answer's claims are truly backed by its cited sources (0..1)."""
    return await _run_judge(
        _prompt_citation_accuracy(answer, sources), "citation_accuracy", llm
    )


# ---------------------------------------------------------------------------
# Composite + consistency helpers
# ---------------------------------------------------------------------------
@dataclass
class AnswerJudgement:
    """All three judges for one answer, plus a composite."""

    correctness: JudgeScore
    faithfulness: JudgeScore
    citation_accuracy: JudgeScore
    composite: float = 0.0
    weights: dict = field(default_factory=lambda: {
        "answer_correctness": 0.5, "faithfulness": 0.3, "citation_accuracy": 0.2,
    })

    def to_dict(self) -> dict:
        return {
            "correctness": self.correctness.to_dict(),
            "faithfulness": self.faithfulness.to_dict(),
            "citation_accuracy": self.citation_accuracy.to_dict(),
            "composite": self.composite,
        }


async def judge_answer(
    query: str,
    answer: str,
    reference_answer: str,
    contexts: list[str],
    sources: list[dict],
    *,
    llm: Optional[SupportsAInvoke] = None,
) -> AnswerJudgement:
    """Run all three judges concurrently and combine into a weighted composite."""
    engine = llm or _default_llm()  # share one instance across the three calls
    correctness, faithfulness, citation = await asyncio.gather(
        judge_answer_correctness(query, answer, reference_answer, llm=engine),
        judge_faithfulness(answer, contexts, llm=engine),
        judge_citation_accuracy(answer, sources, llm=engine),
    )
    j = AnswerJudgement(correctness, faithfulness, citation)
    j.composite = _clamp(
        j.weights["answer_correctness"] * correctness.score
        + j.weights["faithfulness"] * faithfulness.score
        + j.weights["citation_accuracy"] * citation.score
    )
    return j


async def judge_consistency(
    judge_fn: Callable[..., Awaitable[JudgeScore]],
    *args: Any,
    runs: int = 3,
    **kwargs: Any,
) -> dict:
    """Run one judge N times to measure score variance (judge reliability).

    High variance means the judge/prompt is unreliable for that item — a signal to
    tighten the rubric or raise ``temperature=0`` discipline.
    """
    results = await asyncio.gather(*(judge_fn(*args, **kwargs) for _ in range(runs)))
    scores = [r.score for r in results]
    return {
        "scores": scores,
        "mean": statistics.fmean(scores),
        "stdev": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "runs": runs,
    }


def run_judges(
    query: str,
    answer: str,
    reference_answer: str,
    contexts: list[str],
    sources: list[dict],
    *,
    llm: Optional[SupportsAInvoke] = None,
) -> AnswerJudgement:
    """Synchronous convenience wrapper around :func:`judge_answer`."""
    return asyncio.run(
        judge_answer(query, answer, reference_answer, contexts, sources, llm=llm)
    )


# ---------------------------------------------------------------------------
# Offline self-test (no API key required)
# ---------------------------------------------------------------------------
class _FakeLLM:
    """Deterministic stub: returns canned JSON so judges can be tested offline."""

    def __init__(self, payload: str):
        self._payload = payload

    async def ainvoke(self, prompt: str) -> Any:  # noqa: ARG002
        class _Resp:
            content = self._payload
        return _Resp()


def _selftest() -> int:
    """Exercise parsing + judges with a fake LLM. Returns process exit code."""
    print("== _parse_judge_json ==")
    cases = [
        ('{"score": 0.8, "reasoning": "ok"}', 0.8),
        ('```json\n{"score": 1, "reasoning": "x"}\n```', 1.0),
        ('The score is {"score": 0.25, "reasoning": "partial"} overall', 0.25),
        ("0.42", 0.42),                      # bare number fallback
        ('{"score": 5}', 1.0),               # clamp above 1
        ("garbage", None),                   # unparseable
    ]
    ok = True
    for text, expected in cases:
        score, reason = _parse_judge_json(text)
        passed = (score == expected) or (score is None and expected is None)
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {text[:32]!r:35} -> {score} ({reason[:30]})")

    print("\n== judges with fake LLM ==")
    fake = _FakeLLM('{"score": 0.9, "reasoning": "well supported"}')
    j = run_judges(
        query="Do I need approval to clear native vegetation in SA?",
        answer="In South Australia clearing native vegetation needs Native Vegetation Council consent.",
        reference_answer="SA requires approval under the Native Vegetation Act via the Council.",
        contexts=["The Native Vegetation Council administers clearance consent in SA."],
        sources=[{"title": "Native Veg Act", "excerpt": "clearance requires consent"}],
        llm=fake,
    )
    print(json.dumps(j.to_dict(), indent=2))
    ok &= abs(j.composite - 0.9) < 1e-9

    print(f"\nSELF-TEST {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print("Usage: python -m green_gov_rag.eval.llm_judge --selftest")
