"""Hybrid retrieval pipeline: dense + sparse recall, RRF fusion, rerank, routing.

This is the P0 upgrade to GreenGovRAG's retrieval core. Previously retrieval was a
single dense ``similarity_search`` (BM25/RRF/reranker absent, and the geospatial
hybrid path disabled in production). This pipeline wires the full stack:

    query
      -> tiered routing (simple = fast path, complex = full path)
      -> dense recall (vector store, honours metadata filters natively)
      -> sparse recall (BM25, filtered as a constraint layer)
      -> RRF fusion
      -> cross-encoder rerank        (complex queries only, if enabled)
      -> top-k documents with a 0-1 `score` in metadata

Spatial / jurisdiction / ESG metadata filters are applied as a *constraint layer*
on both paths rather than as the ranking signal, preserving GreenGovRAG's domain
model while adding smart recall + precision on top.

The pipeline degrades gracefully: if BM25 or the reranker is unavailable it falls
back to whatever paths are working (down to plain dense search), so a CPU-only
single-node deployment keeps functioning.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain.docstore.document import Document

from green_gov_rag.config import settings
from green_gov_rag.rag.bm25_index import bm25_search
from green_gov_rag.rag.fusion import reciprocal_rank_fusion
from green_gov_rag.rag.query_router import COMPLEX, classify_query_complexity

logger = logging.getLogger(__name__)

# Cap the number of fused candidates fed to the (expensive) reranker.
_RERANK_CANDIDATE_CAP = 30


def _match_metadata(doc: Document, metadata_filters: dict) -> bool:
    """Lenient metadata match for the sparse (BM25) path.

    A document is excluded only when a filter key is present on the document *and*
    does not match. Missing keys are not treated as a mismatch, because BM25 chunk
    metadata is less rich than the dense store's; the dense path is already filtered
    natively, and RRF + rerank + confidence gating handle any residual noise.
    Supports dotted nested keys and list-OR semantics.
    """
    meta = doc.metadata or {}
    for key, expected in metadata_filters.items():
        if key == "region_specified":  # transparency-only, not a real filter
            continue

        # Resolve possibly-nested value (e.g. "esg_metadata.frameworks").
        if "." in key:
            value: object = meta
            for part in key.split("."):
                value = value.get(part) if isinstance(value, dict) else None
                if value is None:
                    break
        else:
            value = meta.get(key)

        if value is None:  # key absent on doc -> lenient, keep
            continue

        if isinstance(expected, list):
            if isinstance(value, list):
                if not any(v in expected for v in value):
                    return False
            elif value not in expected:
                return False
        elif isinstance(value, list):
            if expected not in value:
                return False
        elif value != expected:
            return False
    return True


class HybridRetrievalPipeline:
    """Orchestrates multi-path recall, fusion and reranking over a vector store."""

    def __init__(self, vector_store) -> None:
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        metadata_filters: Optional[dict] = None,
        k: int = 5,
        complexity: Optional[str] = None,
    ) -> list[Document]:
        """Run the hybrid pipeline and return the top-``k`` documents.

        Each returned Document carries a 0-1 relevance ``score`` in metadata
        (rerank score when reranked, otherwise a normalised fusion score).
        """
        recall_k = max(settings.recall_k, k)

        # Tiered routing: decide whether to run the heavy path.
        if complexity is None:
            complexity = (
                classify_query_complexity(query)
                if settings.enable_tiered_routing
                else COMPLEX
            )

        # --- Dense recall (metadata filters applied natively by the store) ---
        try:
            dense = self.vector_store.similarity_search(
                query, k=recall_k, metadata_filters=metadata_filters or None
            )
        except TypeError:
            # Legacy stores without metadata_filters kwarg.
            dense = self.vector_store.similarity_search(query, k=recall_k)
        except Exception as exc:  # noqa: BLE001
            logger.error("Dense retrieval failed: %s", exc)
            dense = []

        # --- Sparse recall (BM25), filtered as a constraint layer ---
        sparse: list[Document] = []
        if settings.enable_bm25:
            sparse_raw = bm25_search(query, recall_k)
            sparse = (
                [d for d in sparse_raw if _match_metadata(d, metadata_filters)]
                if metadata_filters
                else sparse_raw
            )

        # --- RRF fusion ---
        result_lists = [lst for lst in (dense, sparse) if lst]
        if not result_lists:
            return []
        if len(result_lists) == 1:
            fused_docs = list(result_lists[0])
            fused_scores = [None] * len(fused_docs)
        else:
            fused = reciprocal_rank_fusion(result_lists, k=settings.rrf_k)
            fused_docs = [doc for doc, _ in fused]
            fused_scores = [score for _, score in fused]

        # Attach a normalised fusion score (min-max over this result set).
        if fused_scores and fused_scores[0] is not None:
            max_score = max(s for s in fused_scores if s is not None) or 1.0
            annotated: list[Document] = []
            for doc, score in zip(fused_docs, fused_scores):
                norm = (score / max_score) if (score and max_score) else 0.0
                annotated.append(
                    Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, "fused_score": score, "score": norm},
                    )
                )
            fused_docs = annotated

        # --- Rerank (complex queries only when tiered routing is on) ---
        do_rerank = settings.enable_reranker and (
            not settings.enable_tiered_routing or complexity == COMPLEX
        )
        if do_rerank and fused_docs:
            from green_gov_rag.rag.reranker import get_reranker

            reranked = get_reranker().rerank(
                query,
                fused_docs[:_RERANK_CANDIDATE_CAP],
                top_n=max(k, settings.rerank_top_n),
            )
            if reranked:
                return reranked[:k]

        return fused_docs[:k]
