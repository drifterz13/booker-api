from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import ForeignKeyConstraint
from sqlmodel import Field, SQLModel


class BookStatus(StrEnum):
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    FAILED = "failed"


class Book(SQLModel, table=True):
    """Metadata for an uploaded PDF; its bytes live in object storage."""

    __table_args__ = (
        ForeignKeyConstraint(
            ["id", "active_index_id"],
            ["bookindex.book_id", "bookindex.id"],
            name="fk_book_active_index",
            use_alter=True,
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    filename: str
    object_key: str = Field(unique=True)
    checksum: str | None = Field(
        default=None, max_length=64, description="SHA-256, populated during ingestion"
    )
    active_index_id: UUID | None = Field(default=None)
    status: BookStatus = Field(default=BookStatus.UPLOADED)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
