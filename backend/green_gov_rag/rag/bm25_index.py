"""BM25 sparse retrieval path for the hybrid pipeline.

Provides a second (lexical) recall path to complement dense vector search. Dense
search captures semantic similarity; BM25 captures exact terms, legal citations,
acronyms and numeric identifiers that embeddings often blur.

Two backends are supported behind one interface (selected via
``settings.bm25_backend``):

- ``rank_bm25``  : pure-Python in-memory BM25Okapi built from the ``document_chunks``
                   table. Zero external services, CPU-only, ideal for dev / single
                   node. Lazily built and cached; call ``refresh()`` after re-indexing.
- ``elasticsearch``: delegates to an external ES/OpenSearch index for production scale
                   (large corpora, persistence, distributed queries).

The corpus text comes from the PostgreSQL ``Chunk`` table (the same chunks whose
embeddings live in FAISS/Qdrant), so both paths retrieve over the same content.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Optional

from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer (adequate for English regulatory text)."""
    return _TOKEN_RE.findall((text or "").lower())


def _chunk_to_document(chunk) -> Document:
    """Convert a Chunk ORM row into a LangChain Document with citation metadata."""
    meta = dict(chunk.metadata_ or {})
    meta.update(
        {
            "file_id": chunk.file_id,
            "source_id": chunk.source_id,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "page_range": chunk.page_range,
            "section_title": chunk.section_title,
            "section_hierarchy": chunk.section_hierarchy,
            "clause_reference": chunk.clause_reference,
            "source_pdf_url": chunk.source_pdf_url,
            "deep_link": chunk.deep_link,
            "citation": chunk.citation,
        }
    )
    # Drop keys that are None to avoid clobbering values already present in metadata_.
    meta = {k: v for k, v in meta.items() if v is not None}
    return Document(page_content=chunk.text or "", metadata=meta)


class _RankBM25Backend:
    """In-memory BM25Okapi index built lazily from the Chunk table."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bm25 = None
        self._documents: list[Document] = []
        self._tokenized: list[list[str]] = []
        self._built = False

    def _build(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - dependency guard
            msg = "rank_bm25 is required for the in-memory BM25 backend. Install with: pip install rank-bm25"
            raise ImportError(msg) from exc

        from sqlmodel import Session, select

        from green_gov_rag.models import Chunk
        from green_gov_rag.models.base import engine

        documents: list[Document] = []
        with Session(engine) as session:
            for chunk in session.exec(select(Chunk)).all():
                if chunk.text and chunk.text.strip():
                    documents.append(_chunk_to_document(chunk))

        self._documents = documents
        self._tokenized = [_tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        self._built = True
        logger.info("BM25 in-memory index built: %d chunks", len(documents))

    def ensure_built(self) -> None:
        if self._built:
            return
        with self._lock:
            if not self._built:
                self._build()

    def refresh(self) -> None:
        with self._lock:
            self._built = False
            self._build()

    def search(self, query: str, k: int) -> list[Document]:
        self.ensure_built()
        if not self._bm25 or not self._documents:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        # Rank indices by score descending, keep top-k with positive score.
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: list[Document] = []
        for idx in ranked[:k]:
            if scores[idx] <= 0:
                break
            doc = self._documents[idx]
            # Shallow copy metadata so we can attach the sparse score.
            enriched = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "bm25_score": float(scores[idx])},
            )
            results.append(enriched)
        return results


class _ElasticsearchBackend:
    """BM25 via an external Elasticsearch/OpenSearch index (production scale)."""

    def __init__(self, url: str, index: str) -> None:
        self.url = url
        self.index = index
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from elasticsearch import Elasticsearch
            except ImportError as exc:  # pragma: no cover - dependency guard
                msg = "elasticsearch client required for bm25_backend='elasticsearch'. Install with: pip install elasticsearch"
                raise ImportError(msg) from exc
            self._client = Elasticsearch(self.url)
        return self._client

    def ensure_built(self) -> None:  # ES index is managed by the ETL pipeline.
        return None

    def refresh(self) -> None:
        return None

    def search(self, query: str, k: int) -> list[Document]:
        client = self._get_client()
        resp = client.search(
            index=self.index,
            query={"match": {"content": query}},
            size=k,
        )
        results: list[Document] = []
        for hit in resp.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            content = source.pop("content", "")
            results.append(
                Document(
                    page_content=content,
                    metadata={**source, "bm25_score": float(hit.get("_score", 0.0))},
                )
            )
        return results


_backend_singleton = None
_backend_lock = threading.Lock()


def get_bm25_backend():
    """Return the process-wide BM25 backend selected by settings (lazy singleton)."""
    global _backend_singleton
    if _backend_singleton is not None:
        return _backend_singleton
    with _backend_lock:
        if _backend_singleton is None:
            from green_gov_rag.config import settings

            if settings.bm25_backend == "elasticsearch":
                if not settings.elasticsearch_url:
                    msg = "elasticsearch_url must be set when bm25_backend='elasticsearch'"
                    raise ValueError(msg)
                _backend_singleton = _ElasticsearchBackend(
                    settings.elasticsearch_url, settings.elasticsearch_index
                )
            else:
                _backend_singleton = _RankBM25Backend()
    return _backend_singleton


def bm25_search(query: str, k: int = 20) -> list[Document]:
    """Retrieve up to ``k`` chunks for ``query`` via the configured BM25 backend.

    Returns an empty list (never raises) on backend/dependency errors so the hybrid
    pipeline degrades gracefully to dense-only retrieval.
    """
    try:
        return get_bm25_backend().search(query, k)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning("BM25 search unavailable, skipping sparse path: %s", exc)
        return []


def refresh_bm25_index() -> None:
    """Rebuild the in-memory BM25 index (call after re-indexing chunks)."""
    try:
        get_bm25_backend().refresh()
    except Exception as exc:  # noqa: BLE001
        logger.warning("BM25 refresh failed: %s", exc)
