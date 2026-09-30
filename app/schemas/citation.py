from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..models.search import RetrievedPassage

CitationId = Annotated[str, Field(pattern=r"^s[1-9][0-9]*$")]


class CitationSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: CitationId
    book_id: UUID
    index_id: UUID
    chunk_id: UUID
    pdf_pages: list[Annotated[int, Field(strict=True, ge=1)]] = Field(min_length=1)
    section_path: list[str]


class CitationData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[CitationSource]


class CitedPassage(RetrievedPassage):
    citation_id: CitationId


class CitedSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    book_id: UUID
    index_id: UUID
    passages: list[CitedPassage]
