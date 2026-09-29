import hashlib
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

import pymupdf
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from ..models.book import Book, BookStatus
from .storage.storage import ObjectStorage


class BookValidationError(ValueError):
    """The upload does not meet the book upload requirements."""


class BookTooLargeError(BookValidationError):
    """The PDF exceeds the upload limit."""


def validate_book_filename(filename: str) -> str:
    if not filename.strip() or Path(filename).suffix.lower() != ".pdf":
        raise BookValidationError("A PDF filename is required")
    return filename


class BookServiceError(RuntimeError):
    """An external operation failed; retain identifiers for reconciliation."""

    def __init__(self, message: str, *, book_id: UUID, object_key: str) -> None:
        super().__init__(message)
        self.book_id = book_id
        self.object_key = object_key


class BookUploadError(BookServiceError):
    """Object storage could not upload the book."""


class BookPersistenceError(BookServiceError):
    """The database could not save or reload the book's state."""


class BookService:
    """Coordinate upload metadata and PDF storage using a dedicated session.

    The caller owns the session and seekable binary stream. This service commits
    its own writes; supply a session without unrelated pending changes.
    """

    MAX_UPLOAD_BYTES = 100 * 1024 * 1024

    def __init__(
        self,
        *,
        session: Session,
        storage: ObjectStorage,
    ) -> None:
        self._session = session
        self._storage = storage

    def create(self, *, filename: str, source: BinaryIO) -> Book:
        """Validate, hash, and upload a PDF, recording its upload lifecycle.

        A storage failure leaves a failed row. If persistence fails after upload,
        the object may exist with an uploading row; the error carries its ID/key.
        No database transaction is held open during the storage upload.
        """
        checksum = self._validate_upload(filename=filename, source=source)
        book_id = uuid4()
        object_key = f"books/{book_id}.pdf"
        book = Book(
            id=book_id,
            filename=filename,
            object_key=object_key,
            checksum=checksum,
            status=BookStatus.UPLOADING,
        )

        try:
            self._session.add(book)
            self._session.commit()

            # Use the local key: reading expired ORM attributes after commit
            # would start another transaction before the upload.
            try:
                self._storage.upload(source, object_key=object_key)
            except Exception as error:
                book.status = BookStatus.FAILED
                self._session.add(book)
                self._session.commit()
                raise BookUploadError(
                    "Could not upload the book",
                    book_id=book_id,
                    object_key=object_key,
                ) from error

            book.status = BookStatus.UPLOADED
            self._session.add(book)
            self._session.commit()
            self._session.refresh(book)
        except SQLAlchemyError as error:
            self._session.rollback()
            raise BookPersistenceError(
                "Could not persist the book's upload state",
                book_id=book_id,
                object_key=object_key,
            ) from error

        return book

    def list(self, *, offset: int = 0, limit: int = 20) -> list[Book]:
        """List metadata from the database, newest first, including failures."""
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit <= 0:
            raise ValueError("limit must be positive")
        statement = (
            select(Book)
            .order_by(Book.created_at.desc(), Book.id.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(self._session.exec(statement).all())

    def _validate_upload(self, *, filename: str, source: BinaryIO) -> str:
        validate_book_filename(filename)

        source.seek(0)
        content = source.read(self.MAX_UPLOAD_BYTES + 1)
        if not content:
            raise BookValidationError("The PDF is empty")
        if len(content) > self.MAX_UPLOAD_BYTES:
            raise BookTooLargeError(
                f"The PDF exceeds the {self.MAX_UPLOAD_BYTES}-byte upload limit"
            )

        try:
            with pymupdf.open(stream=content, filetype="pdf") as document:
                if not document.is_pdf or document.page_count == 0:
                    raise BookValidationError("The upload must be a PDF with pages")
        except pymupdf.FileDataError as error:
            raise BookValidationError("The upload is not a readable PDF") from error

        source.seek(0)
        return hashlib.sha256(content).hexdigest()
