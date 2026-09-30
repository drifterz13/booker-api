from collections.abc import AsyncIterator
from enum import StrEnum
from uuid import UUID

from fastapi.concurrency import run_in_threadpool
from langchain_core.embeddings import Embeddings
from sqlalchemy.engine import Engine
from sqlmodel import Session

from ....db.vector_store.pgvector import PgVectorStore
from ....models.book import Book
from ....models.book_index import BookIndex, BookIndexStatus
from ....models.search import BookSearchResult, RetrievedPassage, SearchHit


class BookNotFoundError(ValueError):
    pass


class BookNotReadyError(ValueError):
    pass


class SearchPhase(StrEnum):
    EMBEDDING_QUERY = "embedding-query"
    SEARCHING_BOOK = "searching-book"


class BookSearch:
    """Retrieve evidence from the active index captured at request start."""

    def __init__(
        self, *, engine: Engine, embeddings: Embeddings, book_id: UUID
    ) -> None:
        self._engine = engine
        self._embeddings = embeddings
        self.book_id = book_id
        with Session(engine) as session:
            book = session.get(Book, book_id)
            if book is None:
                raise BookNotFoundError("Book not found")
            index = (
                session.get(BookIndex, book.active_index_id)
                if book.active_index_id is not None
                else None
            )
            if (
                index is None
                or index.book_id != book_id
                or index.status != BookIndexStatus.READY
            ):
                raise BookNotReadyError("Book has no ready active index")
            self.index_id = index.id

    def _search(self, vector: list[float]) -> list[SearchHit]:
        with Session(self._engine) as session:
            return PgVectorStore(session).search(vector, index_id=self.index_id)

    async def search(self, query: str) -> BookSearchResult:
        """Return the result when incremental search phases are not needed."""
        async for update in self.stream_search(query):
            if isinstance(update, BookSearchResult):
                return update
        raise RuntimeError("Book search ended without a result")

    async def stream_search(
        self, query: str
    ) -> AsyncIterator[SearchPhase | BookSearchResult]:
        """Yield actual processing phases, followed by retrieved evidence."""
        yield SearchPhase.EMBEDDING_QUERY
        vector = await self._embeddings.aembed_query(query)
        yield SearchPhase.SEARCHING_BOOK
        hits = await run_in_threadpool(self._search, vector)
        yield BookSearchResult(
            book_id=self.book_id,
            index_id=self.index_id,
            passages=[
                RetrievedPassage(
                    chunk_id=hit.chunk_id,
                    text=hit.text,
                    section_path=list(hit.path),
                    pdf_pages=[page + 1 for page in hit.pages],
                )
                for hit in hits
            ],
        )
