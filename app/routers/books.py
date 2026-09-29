from typing import Annotated
from uuid import UUID

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from ..dependencies import BookIngestionServiceDep, BookServiceDep, StorageDep
from ..models.book import Book
from ..models.book_index import BookIndex
from ..schemas.book import (
    BookCreate,
    BookDetail,
    BookListParams,
    BookPdf,
    BookPublic,
    IngestionPublic,
)
from ..services.books import (
    BookAlreadyExistsError,
    BookTooLargeError,
    BookValidationError,
)
from ..services.storage.storage import ObjectStorage

router = APIRouter(prefix="/books", tags=["books"])


def book_public(book: Book, storage: ObjectStorage) -> BookPublic:
    return BookPublic.model_validate(book).model_copy(
        update={
            "thumbnail_url": storage.presign_download(object_key=book.thumbnail_key)
            if book.thumbnail_key
            else None
        }
    )


def book_detail(
    book: Book, index: BookIndex | None, storage: ObjectStorage
) -> BookDetail:
    return BookDetail(
        **book_public(book, storage).model_dump(),
        ingestion=IngestionPublic.model_validate(index) if index else None,
    )


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def create_book(
    payload: BookCreate,
    background_tasks: BackgroundTasks,
    service: BookServiceDep,
    ingestion: BookIngestionServiceDep,
    storage: StorageDep,
) -> BookDetail:
    try:
        book, index = service.create(
            filename=payload.filename, object_key=payload.object_key
        )
    except BookTooLargeError as error:
        raise HTTPException(413, str(error)) from error
    except BookValidationError as error:
        raise HTTPException(400, str(error)) from error
    except BookAlreadyExistsError as error:
        raise HTTPException(409, str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except (BotoCoreError, ClientError) as error:
        raise HTTPException(502, "Could not access upload storage") from error
    except IntegrityError as error:
        if (
            getattr(getattr(error.orig, "diag", None), "constraint_name", None)
            == "book_object_key_key"
        ):
            raise HTTPException(
                409, "The object is already registered as a book"
            ) from error
        raise HTTPException(503, "Could not persist the book") from error
    except SQLAlchemyError as error:
        raise HTTPException(503, "Could not persist the book") from error
    response = book_detail(book, index, storage)
    background_tasks.add_task(ingestion.ingest_in_background, index.id)
    return response


@router.get("")
def list_books(
    params: Annotated[BookListParams, Query()],
    service: BookServiceDep,
    storage: StorageDep,
) -> list[BookPublic]:
    try:
        books = service.list(offset=params.offset, limit=params.limit)
    except SQLAlchemyError as error:
        raise HTTPException(503, "Could not list books") from error
    return [book_public(book, storage) for book in books]


@router.get("/{book_id}")
def get_book(book_id: UUID, service: BookServiceDep, storage: StorageDep) -> BookDetail:
    try:
        book = service.get(book_id)
        if book is None:
            raise HTTPException(404, "Book does not exist")
        index = service.latest_index(book_id)
    except SQLAlchemyError as error:
        raise HTTPException(503, "Could not load the book") from error
    return book_detail(book, index, storage)


@router.get("/{book_id}/pdf")
def get_book_pdf(
    book_id: UUID, service: BookServiceDep, storage: StorageDep
) -> BookPdf:
    """Return a temporary PDF URL, including while indexing is in progress."""
    try:
        book = service.get(book_id)
        if book is None:
            raise HTTPException(404, "Book does not exist")
        storage.size(object_key=book.object_key)
        return BookPdf(url=storage.presign_download(object_key=book.object_key))
    except FileNotFoundError as error:
        raise HTTPException(404, "Book PDF does not exist") from error
    except (BotoCoreError, ClientError) as error:
        raise HTTPException(502, "Could not access PDF storage") from error
    except SQLAlchemyError as error:
        raise HTTPException(503, "Could not load the book") from error
