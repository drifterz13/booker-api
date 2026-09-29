from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from pydantic import AfterValidator
from sqlalchemy.exc import SQLAlchemyError

from ..dependencies import BookIngestionServiceDep, BookServiceDep, BookUploadServiceDep
from ..schemas.book import BookListParams, BookPublic
from ..services.books import (
    BookPersistenceError,
    BookTooLargeError,
    BookUploadError,
    BookValidationError,
    validate_book_filename,
)
from ..services.ingest.books import BookIngestionError
from ..services.ingest.pipeline import BookContentError

router = APIRouter(prefix="/books", tags=["books"])


def validate_pdf_upload(upload: UploadFile) -> UploadFile:
    validate_book_filename(upload.filename or "")
    return upload


PdfUpload = Annotated[UploadFile, File(), AfterValidator(validate_pdf_upload)]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_book(
    file: PdfUpload, uploads: BookUploadServiceDep, ingestion: BookIngestionServiceDep
) -> BookPublic:
    try:
        book = await uploads.create(filename=file.filename or "", source=file.file)
        index_id = await ingestion.ingest(book.id)
    except BookTooLargeError as error:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, detail=str(error)
        ) from error
    except BookValidationError as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except (BookUploadError, BookPersistenceError) as error:
        code = (
            status.HTTP_502_BAD_GATEWAY
            if isinstance(error, BookUploadError)
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        raise HTTPException(
            code,
            detail={"message": str(error), "book_id": str(error.book_id)},
        ) from error
    except BookIngestionError as error:
        code = status.HTTP_502_BAD_GATEWAY
        if isinstance(error.__cause__, BookContentError):
            code = status.HTTP_422_UNPROCESSABLE_CONTENT
        elif isinstance(error.__cause__, SQLAlchemyError):
            code = status.HTTP_503_SERVICE_UNAVAILABLE
        raise HTTPException(
            code,
            detail={
                "message": str(error.__cause__)
                if isinstance(error.__cause__, BookContentError)
                else str(error),
                "book_id": str(error.book_id),
                "index_id": str(error.index_id),
            },
        ) from error
    except SQLAlchemyError as error:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not persist book ingestion",
        ) from error
    return BookPublic.model_validate(book).model_copy(
        update={"active_index_id": index_id}
    )


@router.get("")
def list_books(
    params: Annotated[BookListParams, Query()], service: BookServiceDep
) -> list[BookPublic]:
    try:
        books = service.list(offset=params.offset, limit=params.limit)
    except SQLAlchemyError as error:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail="Could not list books"
        ) from error
    return [BookPublic.model_validate(book) for book in books]
