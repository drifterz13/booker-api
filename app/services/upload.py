import hashlib
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO

from fastapi.concurrency import run_in_threadpool
from sqlalchemy.engine import Engine
from sqlmodel import Session

from ..models import Book
from .books import (
    BookService,
    BookTooLargeError,
    BookValidationError,
    validate_book_filename,
)
from .ingest.pipeline import validate_pdf
from .process_pool import run_in_process_pool
from .storage.storage import ObjectStorage


class BookUploadService:
    """Validate a PDF in a process, then persist its upload in a thread."""

    def __init__(
        self,
        *,
        engine: Engine,
        storage: ObjectStorage,
        process_pool: ProcessPoolExecutor,
    ) -> None:
        self._engine = engine
        self._storage = storage
        self._process_pool = process_pool

    async def create(self, *, filename: str, source: BinaryIO) -> Book:
        validate_book_filename(filename)
        content = await run_in_threadpool(self._read_upload, source)
        await run_in_process_pool(self._process_pool, validate_pdf, content)
        checksum = await run_in_threadpool(hashlib.sha256, content)
        return await run_in_threadpool(
            self._persist, filename, source, checksum.hexdigest()
        )

    @staticmethod
    def _read_upload(source: BinaryIO) -> bytes:
        source.seek(0)
        content = source.read(BookService.MAX_UPLOAD_BYTES + 1)
        source.seek(0)
        if not content:
            raise BookValidationError("The PDF is empty")
        if len(content) > BookService.MAX_UPLOAD_BYTES:
            raise BookTooLargeError(
                f"The PDF exceeds the {BookService.MAX_UPLOAD_BYTES}-byte upload limit"
            )
        return content

    def _persist(self, filename: str, source: BinaryIO, checksum: str) -> Book:
        with Session(self._engine) as session:
            return BookService(session=session, storage=self._storage).create(
                filename=filename, source=source, checksum=checksum
            )
