from pathlib import Path
from uuid import UUID

from sqlmodel import Session, select

from ..models.book import Book
from ..models.book_index import BookIndex
from .ingest.pipeline import PIPELINE_VERSION
from .storage.storage import ObjectStorage


class BookValidationError(ValueError):
    """The object does not meet the book requirements."""


class BookTooLargeError(BookValidationError):
    """The PDF exceeds the upload limit."""


class BookAlreadyExistsError(ValueError):
    """The object is already registered as a book."""


def validate_book_filename(filename: str) -> str:
    if not filename.strip() or Path(filename).suffix.lower() != ".pdf":
        raise BookValidationError("A PDF filename is required")
    return filename


class BookService:
    """Register completed objects and read book metadata."""

    MAX_UPLOAD_BYTES = 100 * 1024 * 1024

    def __init__(self, *, session: Session, storage: ObjectStorage) -> None:
        self._session = session
        self._storage = storage

    def create(self, *, filename: str, object_key: str) -> tuple[Book, BookIndex]:
        """Commit the book and its building index before scheduling ingestion."""
        validate_book_filename(filename)
        size = self._storage.size(object_key=object_key)
        if size == 0:
            raise BookValidationError("The PDF is empty")
        if size > self.MAX_UPLOAD_BYTES:
            raise BookTooLargeError(
                f"The PDF exceeds the {self.MAX_UPLOAD_BYTES}-byte upload limit"
            )
        if self._session.exec(
            select(Book).where(Book.object_key == object_key)
        ).first():
            raise BookAlreadyExistsError("The object is already registered as a book")
        book = Book(filename=filename, object_key=object_key)
        index = BookIndex(book_id=book.id, pipeline_version=PIPELINE_VERSION)
        self._session.add(book)
        self._session.flush()
        self._session.add(index)
        self._session.commit()
        self._session.refresh(book)
        self._session.refresh(index)
        return book, index

    def get(self, book_id: UUID) -> Book | None:
        return self._session.get(Book, book_id)

    def latest_index(self, book_id: UUID) -> BookIndex | None:
        return self._session.exec(
            select(BookIndex)
            .where(BookIndex.book_id == book_id)
            .order_by(BookIndex.created_at.desc(), BookIndex.id.desc())
            .limit(1)
        ).first()

    def list(self, *, offset: int = 0, limit: int = 20) -> list[Book]:
        if offset < 0 or limit <= 0:
            raise ValueError("Invalid pagination")
        return list(
            self._session.exec(
                select(Book)
                .order_by(Book.created_at.desc(), Book.id.desc())
                .offset(offset)
                .limit(limit)
            ).all()
        )
