from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel


class BookStatus(StrEnum):
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    FAILED = "failed"


class Book(SQLModel, table=True):
    """Metadata for an uploaded PDF; its bytes live in object storage."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    filename: str
    object_key: str = Field(unique=True)
    checksum: str = Field(max_length=64, description="SHA-256 of the PDF bytes")
    status: BookStatus = Field(default=BookStatus.UPLOADED)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
