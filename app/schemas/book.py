from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from ..models.book import BookStatus
from ..models.book_index import BookIndexStatus
from ..services.books import validate_book_filename
from .upload import ObjectKey


class BookCreate(BaseModel):
    filename: Annotated[
        str, Field(max_length=255), AfterValidator(validate_book_filename)
    ]
    object_key: ObjectKey


class BookPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    status: BookStatus
    created_at: datetime
    active_index_id: UUID | None = None
    thumbnail_url: str | None = None


class BookListParams(BaseModel):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)


class IngestionPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: BookIndexStatus


class BookDetail(BookPublic):
    ingestion: IngestionPublic | None = None


class BookPdf(BaseModel):
    url: str
    expires_in: int = 900
