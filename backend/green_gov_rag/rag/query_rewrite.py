"""LLM query rewriting for multi-turn coreference resolution.

In a multi-turn conversation, follow-up questions are often not self-contained:

    User: What are the native vegetation clearing rules in South Australia?
    User: Do I need a permit for *it* on rural land?

Retrieval over the second query alone is poor because "it" and the SA context are
missing. This module rewrites a follow-up into a standalone query using the recent
conversation history, so the retrieval pipeline gets a complete, unambiguous query.

If there is no history, the LLM is unavailable, or the rewrite looks degenerate, the
original query is returned unchanged (safe no-op).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """You rewrite a user's latest question into a self-contained search query.

Use the conversation so far to resolve pronouns and implicit references (e.g. "it",
"that", "there") and to carry over context like jurisdiction, region or topic that
the latest message omits. Do NOT answer the question. Do NOT add information that was
never mentioned. Return ONLY the rewritten query on a single line.

Conversation so far:
{history}

Latest user question: {query}

Rewritten standalone query:"""


def rewrite_query(query: str, history_text: str) -> str:
    """Rewrite ``query`` into a standalone form using ``history_text``.

    Returns the original query unchanged when there's no history or on any error.
    """
    if not history_text or not history_text.strip():
        return query

    try:
        from langchain.schema import HumanMessage

        from green_gov_rag.rag.llm_factory import get_llm

        llm = get_llm(temperature=0.0, max_tokens=120)
        prompt = _REWRITE_PROMPT.format(history=history_text, query=query)
        response = llm.invoke([HumanMessage(content=prompt)])
        rewritten = (
            response.content if hasattr(response, "content") else str(response)
        ).strip()

        # Guard against degenerate rewrites (empty, echoed label, or absurdly long).
        if not rewritten or len(rewritten) > 4 * len(query) + 200:
            return query
        # Strip any accidental leading label.
        for prefix in ("Rewritten standalone query:", "Query:", "Rewritten query:"):
            if rewritten.lower().startswith(prefix.lower()):
                rewritten = rewritten[len(prefix) :].strip()
        return rewritten or query
    except Exception as exc:  # noqa: BLE001 - safe no-op on failure
        logger.warning("Query rewrite failed (%s); using original query.", exc)
        return query
