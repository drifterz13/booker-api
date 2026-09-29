import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from fastapi.concurrency import run_in_threadpool
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session

from ...db.vector_store.pgvector import PgVectorStore
from ...models import Book, BookIndex, EmbeddedChunk
from ...models.book import BookStatus
from ...models.book_index import BookIndexStatus
from ..chunker import ChunkEmbedder
from ..process_pool import run_in_process_pool
from ..storage.storage import ObjectStorage
from .pipeline import PIPELINE_VERSION, chunk_book, extract_book


@dataclass(frozen=True, slots=True)
class PendingIndex:
    id: UUID
    book_id: UUID
    object_key: str


class BookIngestionError(RuntimeError):
    """Ingestion failed after creating an index version."""

    def __init__(self, *, book_id: UUID, index_id: UUID) -> None:
        super().__init__("Could not ingest the book")
        self.book_id = book_id
        self.index_id = index_id


class BookIngestionService:
    """Extract, chunk, embed, and activate an uploaded book's index.

    The caller owns the engine, storage, process pool, and embedding client.
    The embedder must use the model and dimensions defined in BookIndex.
    Each database operation uses its own session and transaction.
    """

    def __init__(
        self,
        *,
        engine: Engine,
        storage: ObjectStorage,
        process_pool: ProcessPoolExecutor,
        embedder: ChunkEmbedder,
    ) -> None:
        self._engine = engine
        self._storage = storage
        self._process_pool = process_pool
        self._embedder = embedder

    async def ingest(self, book_id: UUID) -> UUID:
        """Create a new version and return its ID after successful activation."""
        index = await run_in_threadpool(self._create_index, book_id)
        try:
            with TemporaryDirectory(prefix="booker-ingest-") as directory:
                source = Path(directory) / "book.pdf"
                await run_in_threadpool(
                    self._storage.download,
                    object_key=index.object_key,
                    destination=source,
                )
                segments = await run_in_process_pool(
                    self._process_pool, extract_book, source
                )
                chunks = await run_in_process_pool(
                    self._process_pool, chunk_book, segments
                )
                items = await self._embedder.aembed(chunks)
                await run_in_threadpool(self._activate_index, index, items)
        except (Exception, asyncio.CancelledError) as error:
            try:
                await run_in_threadpool(self._mark_failed, index.id)
            except SQLAlchemyError:
                error.add_note("Could not persist the failed ingestion status")
            if isinstance(error, asyncio.CancelledError):
                raise
            raise BookIngestionError(book_id=book_id, index_id=index.id) from error
        return index.id

    def _create_index(self, book_id: UUID) -> PendingIndex:
        with Session(self._engine) as session:
            book = session.get(Book, book_id)
            if book is None:
                raise ValueError("Book does not exist")
            if book.status != BookStatus.UPLOADED:
                raise ValueError("Only uploaded books can be ingested")
            index = BookIndex(book_id=book_id, pipeline_version=PIPELINE_VERSION)
            pending = PendingIndex(index.id, book_id, book.object_key)
            session.add(index)
            session.commit()
            return pending

    def _activate_index(
        self, pending: PendingIndex, items: list[EmbeddedChunk]
    ) -> None:
        with Session(self._engine) as session, session.begin():
            # Lock the book first to serialize activation of different versions.
            book = session.get(Book, pending.book_id, with_for_update=True)
            if book is None or book.status != BookStatus.UPLOADED:
                raise ValueError("The uploaded book is no longer available")
            count = PgVectorStore(session).add_chunks(pending.id, items)
            index = session.get(BookIndex, pending.id)
            if index is None:
                raise ValueError("Book index does not exist")
            index.chunk_count = count
            index.status = BookIndexStatus.READY
            book.active_index_id = pending.id
            session.add_all([index, book])

    def _mark_failed(self, index_id: UUID) -> None:
        with Session(self._engine) as session, session.begin():
            index = session.get(BookIndex, index_id, with_for_update=True)
            if index is not None and index.status == BookIndexStatus.BUILDING:
                index.status = BookIndexStatus.FAILED
                session.add(index)
