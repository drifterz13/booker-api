from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlmodel import Field, SQLModel

from .book import Book  # noqa: F401 - register the referenced book table

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


class BookIndexStatus(StrEnum):
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


class BookIndex(SQLModel, table=True):
    """One processing version of a book; ready versions are retained unchanged.

    Activation and lifecycle transitions belong to the ingestion service.
    """

    __table_args__ = (
        UniqueConstraint("book_id", "id", name="uq_bookindex_book_id_id"),
        CheckConstraint("chunk_count >= 0", name="ck_bookindex_chunk_count"),
        CheckConstraint(
            f"embedding_dimensions = {EMBEDDING_DIMENSIONS}",
            name="ck_bookindex_embedding_dimensions",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    book_id: UUID = Field(foreign_key="book.id", index=True)
    embedding_model: str = Field(default=EMBEDDING_MODEL)
    embedding_dimensions: int = Field(
        default=EMBEDDING_DIMENSIONS,
        ge=EMBEDDING_DIMENSIONS,
        le=EMBEDDING_DIMENSIONS,
    )
    pipeline_version: str = Field(min_length=1)
    status: BookIndexStatus = Field(default=BookIndexStatus.BUILDING)
    chunk_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
