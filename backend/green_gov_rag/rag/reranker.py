"""Cross-encoder reranker for precision reordering of fused candidates.

A bi-encoder (the embedding model) is fast but coarse; a cross-encoder jointly
encodes (query, passage) and scores relevance directly, which is markedly more
accurate for the final top-k. This is the "Reranker" stage of the pipeline:

    dense + sparse -> RRF fuse -> [Reranker] -> top-N -> LLM

The reranker is expensive (~100ms+), so it is controlled by tiered routing (only
run for complex queries) and can be disabled entirely via settings. If the model
or ``sentence-transformers`` is unavailable, reranking is skipped and the fused
order is preserved (graceful degradation, CPU-only path stays functional).

Raw cross-encoder outputs are logits; we map them to a 0-1 range with a sigmoid so
downstream confidence gating has a stable, interpretable score.
"""

from __future__ import annotations

import logging
import math
import threading

from langchain.docstore.document import Document

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:  # pragma: no cover - extreme logits
        return 0.0 if x < 0 else 1.0


class Reranker:
    """Lazy-loading cross-encoder reranker (process-wide singleton via get_reranker)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        self._unavailable = False

    def _get_model(self):
        if self._model is not None or self._unavailable:
            return self._model
        with self._lock:
            if self._model is None and not self._unavailable:
                try:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self.model_name)
                    logger.info("Reranker model loaded: %s", self.model_name)
                except Exception as exc:  # noqa: BLE001 - degrade gracefully
                    logger.warning(
                        "Reranker unavailable (%s); skipping rerank stage.", exc
                    )
                    self._unavailable = True
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int,
    ) -> list[Document]:
        """Reorder ``documents`` by cross-encoder relevance and keep top-N.

        Each returned Document gets ``metadata['rerank_score']`` (0-1) and
        ``metadata['score']`` set to the same value for downstream consumers.
        On any failure the input order is preserved (truncated to top_n).
        """
        if not documents:
            return []

        model = self._get_model()
        if model is None:
            return documents[:top_n]

        try:
            pairs = [(query, doc.page_content or "") for doc in documents]
            raw_scores = model.predict(pairs)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning("Rerank prediction failed (%s); keeping fused order.", exc)
            return documents[:top_n]

        scored: list[tuple[Document, float]] = []
        for doc, raw in zip(documents, raw_scores):
            norm = _sigmoid(float(raw))
            enriched = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "rerank_score": norm,
                    "score": norm,
                },
            )
            scored.append((enriched, norm))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in scored[:top_n]]


_reranker_singleton: Reranker | None = None
_singleton_lock = threading.Lock()


def get_reranker() -> Reranker:
    """Return the process-wide reranker instance (model loaded on first use)."""
    global _reranker_singleton
    if _reranker_singleton is None:
        with _singleton_lock:
            if _reranker_singleton is None:
                from green_gov_rag.config import settings

                _reranker_singleton = Reranker(settings.reranker_model)
    return _reranker_singleton
