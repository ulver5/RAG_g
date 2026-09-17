"""add bad_cases table

Revision ID: a1b2c3d4e5f6
Revises: d7ed7ff5e711
Create Date: 2026-08-12 00:00:00.000000

"""
import sqlalchemy as sa
import sqlmodel

from alembic import op

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "d7ed7ff5e711"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "bad_cases",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("query_id", sa.Integer(), nullable=True),
        sa.Column("session_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("query_text", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("answer_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column("filters_applied", sa.JSON(), nullable=True),
        sa.Column("source_count", sa.Integer(), nullable=False),
        sa.Column("top_source_titles", sa.JSON(), nullable=True),
        sa.Column("category", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("resolved", sa.Boolean(), nullable=False),
        sa.Column("review_notes", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_bad_cases_query_id"), "bad_cases", ["query_id"], unique=False)
    op.create_index(
        op.f("ix_bad_cases_session_id"), "bad_cases", ["session_id"], unique=False
    )
    op.create_index(
        op.f("ix_bad_cases_answer_type"), "bad_cases", ["answer_type"], unique=False
    )
    op.create_index(
        op.f("ix_bad_cases_category"), "bad_cases", ["category"], unique=False
    )
    op.create_index(
        op.f("ix_bad_cases_resolved"), "bad_cases", ["resolved"], unique=False
    )
    op.create_index(
        op.f("ix_bad_cases_created_at"), "bad_cases", ["created_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_bad_cases_created_at"), table_name="bad_cases")
    op.drop_index(op.f("ix_bad_cases_resolved"), table_name="bad_cases")
    op.drop_index(op.f("ix_bad_cases_category"), table_name="bad_cases")
    op.drop_index(op.f("ix_bad_cases_answer_type"), table_name="bad_cases")
    op.drop_index(op.f("ix_bad_cases_session_id"), table_name="bad_cases")
    op.drop_index(op.f("ix_bad_cases_query_id"), table_name="bad_cases")
    op.drop_table("bad_cases")
