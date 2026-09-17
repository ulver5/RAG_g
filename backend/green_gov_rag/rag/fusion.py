"""Reciprocal Rank Fusion (RRF) for combining multiple ranked retrieval results.

RRF is a rank-based fusion method that is robust to differing score scales across
retrievers (e.g. dense cosine similarity vs. BM25 term scores). It only uses the
*rank* of a document within each result list, so no score normalisation is needed.

    RRF(d) = sum over retrievers i of  1 / (k + rank_i(d))

where rank_i(d) is the 1-based rank of document d in retriever i's result list
(documents absent from a list simply do not contribute a term).

This replaces GreenGovRAG's previous heuristic 3:1 jurisdiction interleave as the
primary fusion mechanism (that heuristic is retained downstream as a soft boost).
"""

from __future__ import annotations

import hashlib
from typing import Callable

from langchain.docstore.document import Document


def default_doc_key(doc: Document) -> str:
    """Stable identity for a document across retrieval paths.

    Dense (vector store) and sparse (BM25) results carry different metadata but the
    same chunk ``page_content``. Prefer an explicit id, then file_id+chunk index,
    then fall back to a hash of the content so the same chunk fuses correctly
    regardless of which path surfaced it.
    """
    meta = getattr(doc, "metadata", {}) or {}
    for id_key in ("id", "_id", "chunk_id"):
        if meta.get(id_key) is not None:
            return f"id:{meta[id_key]}"

    file_id = meta.get("file_id")
    chunk_index = meta.get("chunk_index")
    if file_id is not None and chunk_index is not None:
        return f"file:{file_id}:{chunk_index}"

    content = (doc.page_content or "").strip()
    return "sha:" + hashlib.md5(content.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    *,
    k: int = 60,
    key_fn: Callable[[Document], str] = default_doc_key,
    weights: list[float] | None = None,
) -> list[tuple[Document, float]]:
    """Fuse several ranked lists of Documents using Reciprocal Rank Fusion.

    Args:
        result_lists: One ranked list of Documents per retriever (best first).
        k: RRF constant. Larger k flattens the contribution of top ranks.
        key_fn: Function mapping a Document to a stable identity string.
        weights: Optional per-retriever weights (defaults to 1.0 each).

    Returns:
        List of ``(document, rrf_score)`` tuples sorted by score descending.
        The Document instance kept is the first one seen for a given key.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        msg = "weights length must match number of result lists"
        raise ValueError(msg)

    scores: dict[str, float] = {}
    first_seen: dict[str, Document] = {}

    for retriever_idx, docs in enumerate(result_lists):
        weight = weights[retriever_idx]
        for rank, doc in enumerate(docs, start=1):
            key = key_fn(doc)
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (k + rank))
            if key not in first_seen:
                first_seen[key] = doc

    fused = [(first_seen[key], score) for key, score in scores.items()]
    fused.sort(key=lambda pair: pair[1], reverse=True)
    return fused
