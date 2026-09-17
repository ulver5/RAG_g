"""Evaluation dataset schema and relevance matching.

A dataset is a JSON file: a list of items, each describing a query and its ground
truth. Ground truth can be given as relevant file ids (preferred, precise) and/or
expected keywords/phrases that a correct source chunk should contain.

Example ``eval_dataset.json``::

    [
      {
        "id": "sa-veg-clearing-permit",
        "query": "Do I need a permit to clear native vegetation in South Australia?",
        "filters": {"region": "South Australia"},
        "relevant_file_ids": ["sa_native_veg_act_2003"],
        "expected_keywords": ["native vegetation council", "clearance"],
        "reference_answer": "In SA, clearing native vegetation generally requires ..."
      }
    ]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class EvalItem:
    """A single evaluation query with ground truth."""

    id: str
    query: str
    filters: dict[str, Any] = field(default_factory=dict)
    relevant_file_ids: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    reference_answer: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "EvalItem":
        return cls(
            id=str(data.get("id") or data.get("query", "")[:40]),
            query=data["query"],
            filters=data.get("filters") or {},
            relevant_file_ids=[str(x) for x in (data.get("relevant_file_ids") or [])],
            expected_keywords=[str(x) for x in (data.get("expected_keywords") or [])],
            reference_answer=data.get("reference_answer"),
        )


def load_dataset(path: str | Path) -> list[EvalItem]:
    """Load an evaluation dataset from a JSON file."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        msg = "Evaluation dataset must be a JSON list of items"
        raise ValueError(msg)
    return [EvalItem.from_dict(d) for d in raw]


def _doc_text(doc: Any) -> str:
    if hasattr(doc, "page_content"):
        return doc.page_content or ""
    if isinstance(doc, dict):
        return doc.get("excerpt") or doc.get("content") or ""
    return getattr(doc, "excerpt", "") or getattr(doc, "content", "") or ""


def _doc_file_id(doc: Any) -> Optional[str]:
    meta = getattr(doc, "metadata", None)
    if meta is None and isinstance(doc, dict):
        meta = doc.get("metadata", {})
        # dict sources may also carry file_id at top level
        return str(doc.get("file_id") or (meta or {}).get("file_id") or "") or None
    if meta:
        fid = meta.get("file_id")
        return str(fid) if fid is not None else None
    return None


def item_is_hit(item: EvalItem, retrieved_docs: list) -> bool:
    """True if any retrieved doc satisfies the item's ground truth.

    A doc is a hit if its ``file_id`` is in ``relevant_file_ids``, OR (when keywords
    are given) if it contains all/any expected keywords. Keyword matching requires at
    least one expected keyword to appear (case-insensitive substring).
    """
    if not retrieved_docs:
        return False

    if item.relevant_file_ids:
        rel = set(item.relevant_file_ids)
        for doc in retrieved_docs:
            fid = _doc_file_id(doc)
            if fid and fid in rel:
                return True

    if item.expected_keywords:
        keywords = [k.lower() for k in item.expected_keywords]
        for doc in retrieved_docs:
            text = _doc_text(doc).lower()
            if any(k in text for k in keywords):
                return True

    return False
