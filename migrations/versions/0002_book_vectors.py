"""Add book index versions, vector chunks, HNSW, and active index selection."""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import VECTOR
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "bookindex",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("book_id", sa.Uuid(), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("pipeline_version", sa.String(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("BUILDING", "READY", "FAILED", name="bookindexstatus"),
            nullable=False,
        ),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["book_id"], ["book.id"]),
        sa.UniqueConstraint("book_id", "id", name="uq_bookindex_book_id_id"),
        sa.CheckConstraint("chunk_count >= 0", name="ck_bookindex_chunk_count"),
        sa.CheckConstraint(
            "embedding_dimensions = 1536", name="ck_bookindex_embedding_dimensions"
        ),
    )
    op.create_index("ix_bookindex_book_id", "bookindex", ["book_id"])
    op.create_table(
        "bookchunk",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("index_id", sa.Uuid(), nullable=False),
        sa.Column("segment_index", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("section_path", postgresql.JSONB(), nullable=False),
        sa.Column("pdf_pages", postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.Column("embedding", VECTOR(1536), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["index_id"], ["bookindex.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "index_id", "segment_index", "chunk_index", name="uq_bookchunk_position"
        ),
        sa.CheckConstraint("segment_index >= 0", name="ck_bookchunk_segment_index"),
        sa.CheckConstraint("chunk_index >= 0", name="ck_bookchunk_chunk_index"),
        sa.CheckConstraint(
            "cardinality(pdf_pages) > 0 AND 0 <= ALL(pdf_pages)",
            name="ck_bookchunk_pdf_pages",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(section_path) = 'array'", name="ck_bookchunk_section_path"
        ),
    )
    op.create_index("ix_bookchunk_index_id", "bookchunk", ["index_id"])
    op.create_index(
        "ix_bookchunk_embedding_hnsw",
        "bookchunk",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.add_column("book", sa.Column("active_index_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_book_active_index",
        "book",
        "bookindex",
        ["id", "active_index_id"],
        ["book_id", "id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_book_active_index", "book", type_="foreignkey")
    op.drop_column("book", "active_index_id")
    op.drop_table("bookchunk")
    op.drop_table("bookindex")
    sa.Enum(name="bookindexstatus").drop(op.get_bind(), checkfirst=True)
    # The extension may be shared by other applications; retain it.
