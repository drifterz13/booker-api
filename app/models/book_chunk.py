from uuid import UUID, uuid4

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import CheckConstraint, Index, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlmodel import Field, SQLModel

from .book_index import EMBEDDING_DIMENSIONS


class BookChunk(SQLModel, table=True):
    """A passage and its embedding within one book index version.

    PDF pages use zero-based physical page indices, matching the extractor and
    in-memory Chunk model. Convert to one-based numbers when presenting citations.
    """

    __table_args__ = (
        UniqueConstraint(
            "index_id", "segment_index", "chunk_index", name="uq_bookchunk_position"
        ),
        CheckConstraint("segment_index >= 0", name="ck_bookchunk_segment_index"),
        CheckConstraint("chunk_index >= 0", name="ck_bookchunk_chunk_index"),
        CheckConstraint(
            "cardinality(pdf_pages) > 0 AND 0 <= ALL(pdf_pages)",
            name="ck_bookchunk_pdf_pages",
        ),
        CheckConstraint(
            "jsonb_typeof(section_path) = 'array'", name="ck_bookchunk_section_path"
        ),
        Index(
            "ix_bookchunk_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    index_id: UUID = Field(foreign_key="bookindex.id", ondelete="CASCADE", index=True)
    segment_index: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    content: str = Field(min_length=1, sa_type=Text)
    section_path: list[str] = Field(sa_type=JSONB)
    pdf_pages: list[int] = Field(min_length=1, sa_type=ARRAY(Integer))
    embedding: list[float] = Field(
        min_length=EMBEDDING_DIMENSIONS,
        max_length=EMBEDDING_DIMENSIONS,
        sa_type=VECTOR(EMBEDDING_DIMENSIONS),
    )
