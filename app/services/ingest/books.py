import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from io import BytesIO
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
from ..books import BookService, BookTooLargeError
from ..chunker import ChunkEmbedder
from ..process_pool import run_in_process_pool
from ..storage.storage import ObjectStorage
from ..thumbnail import render_thumbnail
from .pipeline import PIPELINE_VERSION, chunk_book, extract_book, verify_pdf

logger = logging.getLogger(__name__)


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
        return await self._ingest(index)

    async def ingest_index(self, index_id: UUID) -> UUID:
        """Process the building index already committed by book creation."""
        index = await run_in_threadpool(self._load_index, index_id)
        return await self._ingest(index)

    async def ingest_in_background(self, index_id: UUID) -> None:
        """Persist failures and log them after the HTTP response is sent."""
        try:
            await self.ingest_index(index_id)
        except Exception:
            logger.exception("Background ingestion failed for index %s", index_id)

    def _load_index(self, index_id: UUID) -> PendingIndex:
        with Session(self._engine) as session:
            index = session.get(BookIndex, index_id)
            if index is None or index.status != BookIndexStatus.BUILDING:
                raise ValueError("Index is not available for ingestion")
            book = session.get(Book, index.book_id)
            if book is None or book.status != BookStatus.UPLOADED:
                raise ValueError("The uploaded book is not available")
            return PendingIndex(index.id, book.id, book.object_key)

    async def _ingest(self, index: PendingIndex) -> UUID:
        try:
            with TemporaryDirectory(prefix="booker-ingest-") as directory:
                source = Path(directory) / "book.pdf"
                await run_in_threadpool(
                    self._storage.download,
                    object_key=index.object_key,
                    destination=source,
                )
                metadata = await run_in_threadpool(source.stat)
                if metadata.st_size > BookService.MAX_UPLOAD_BYTES:
                    raise BookTooLargeError(
                        "The downloaded PDF exceeds the upload limit"
                    )
                checksum = await run_in_process_pool(
                    self._process_pool, verify_pdf, source
                )
                await self._create_thumbnail(index.book_id, source)
                segments = await run_in_process_pool(
                    self._process_pool, extract_book, source
                )
                chunks = await run_in_process_pool(
                    self._process_pool, chunk_book, segments
                )
                items = await self._embedder.aembed(chunks)
                await run_in_threadpool(self._activate_index, index, items, checksum)
        except (Exception, asyncio.CancelledError) as error:
            try:
                await run_in_threadpool(self._mark_failed, index.id)
            except SQLAlchemyError:
                error.add_note("Could not persist the failed ingestion status")
            if isinstance(error, asyncio.CancelledError):
                raise
            raise BookIngestionError(
                book_id=index.book_id, index_id=index.id
            ) from error
        return index.id

    async def _create_thumbnail(self, book_id: UUID, source: Path) -> None:
        try:
            content = await run_in_process_pool(
                self._process_pool, render_thumbnail, source
            )
            key = f"thumbnails/{book_id}.png"
            await run_in_threadpool(
                self._storage.upload,
                BytesIO(content),
                object_key=key,
                content_type="image/png",
            )
            await run_in_threadpool(self._save_thumbnail_key, book_id, key)
        except Exception:
            logger.exception("Could not create thumbnail for book %s", book_id)

    def _save_thumbnail_key(self, book_id: UUID, key: str) -> None:
        with Session(self._engine) as session, session.begin():
            book = session.get(Book, book_id, with_for_update=True)
            if book is None:
                raise ValueError("Book does not exist")
            book.thumbnail_key = key
            session.add(book)

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
        self, pending: PendingIndex, items: list[EmbeddedChunk], checksum: str
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
            book.checksum = checksum
            session.add_all([index, book])

    def _mark_failed(self, index_id: UUID) -> None:
        with Session(self._engine) as session, session.begin():
            index = session.get(BookIndex, index_id, with_for_update=True)
            if index is not None and index.status == BookIndexStatus.BUILDING:
                index.status = BookIndexStatus.FAILED
                session.add(index)
