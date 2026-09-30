from uuid import UUID

from ...models.search import BookSearchResult
from ...schemas.citation import (
    CitationData,
    CitationSource,
    CitedPassage,
    CitedSearchResult,
)


class CitationRegistry:
    """Authoritative sources for one response; never populated from client history."""

    def __init__(self) -> None:
        self._sources: dict[str, CitationSource] = {}
        # Chunk IDs are globally unique primary keys, including across indexes.
        self._ids_by_chunk: dict[UUID, str] = {}

    def register(self, result: BookSearchResult) -> CitedSearchResult:
        passages = []
        for passage in result.passages:
            citation_id = self._ids_by_chunk.get(passage.chunk_id)
            if citation_id is None:
                citation_id = f"s{len(self._sources) + 1}"
                self._ids_by_chunk[passage.chunk_id] = citation_id
                self._sources[citation_id] = CitationSource(
                    id=citation_id,
                    book_id=result.book_id,
                    index_id=result.index_id,
                    chunk_id=passage.chunk_id,
                    pdf_pages=passage.pdf_pages,
                    section_path=passage.section_path,
                )
            passages.append(
                CitedPassage(**passage.model_dump(), citation_id=citation_id)
            )
        return CitedSearchResult(
            book_id=result.book_id, index_id=result.index_id, passages=passages
        )

    def snapshot(self) -> CitationData:
        return CitationData(sources=list(self._sources.values()))
