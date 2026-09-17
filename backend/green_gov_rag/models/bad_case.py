"""Bad-case model for the metric-driven improvement loop.

Low-confidence answers and clarification fallbacks are persisted here so they can be
reviewed weekly, categorised (retrieval failure / chunking / knowledge gap), and used
to drive knowledge-base, chunking and graph iteration. This is the "bad case 库"
closing the feedback loop that GreenGovRAG previously lacked.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import JSON, Column, Field, SQLModel


class BadCase(SQLModel, table=True):
    """A query whose retrieval confidence was low (or that was gated to clarify)."""

    __tablename__ = "bad_cases"

    id: Optional[int] = Field(default=None, primary_key=True)

    # Link back to the query history row, if available.
    query_id: Optional[int] = Field(
        default=None, index=True, description="Related QueryHistory.id"
    )
    session_id: Optional[str] = Field(default=None, index=True)

    query_text: str = Field(description="The user query that triggered the bad case")
    answer_type: str = Field(
        description="Gate outcome: 'clarification' or 'partial'", index=True
    )
    confidence_score: float = Field(description="Top retrieval confidence (0-1)")

    filters_applied: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    source_count: int = Field(default=0, description="Number of sources retrieved")
    top_source_titles: Optional[list[str]] = Field(
        default=None, sa_column=Column(JSON)
    )

    # Filled in during weekly review.
    category: Optional[str] = Field(
        default=None,
        index=True,
        description="Reviewer label: retrieval_failure / chunking / knowledge_gap / other",
    )
    resolved: bool = Field(default=False, index=True)
    review_notes: Optional[str] = Field(default=None)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), index=True
    )
