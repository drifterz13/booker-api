from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RetrievedPassage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: UUID
    text: str
    section_path: list[str]
    pdf_pages: list[Annotated[int, Field(strict=True, ge=1)]] = Field(min_length=1)
    """One-based physical PDF pages, independent of printed page labels."""


class BookSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    book_id: UUID
    index_id: UUID
    passages: list[RetrievedPassage]


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A retrieved passage, independent of its storage backend or local PDF path."""

    chunk_id: UUID
    index_id: UUID
    text: str
    path: tuple[str, ...]
    pages: tuple[int, ...]
    distance: float

    @property
    def start_page(self) -> int:
        return self.pages[0] + 1

    @property
    def end_page(self) -> int:
        return self.pages[-1] + 1
