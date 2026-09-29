from typing import Annotated

from fastapi import Depends, Request
from sqlmodel import Session

from .db.database import get_session
from .services.books import BookService
from .services.chunker import ChunkEmbedder
from .services.ingest.books import BookIngestionService
from .services.storage.storage import ObjectStorage
from .services.upload import BookUploadService

SessionDep = Annotated[Session, Depends(get_session)]


def get_storage(request: Request) -> ObjectStorage:
    return request.app.state.storage


StorageDep = Annotated[ObjectStorage, Depends(get_storage)]


def get_book_service(session: SessionDep, storage: StorageDep) -> BookService:
    return BookService(session=session, storage=storage)


BookServiceDep = Annotated[BookService, Depends(get_book_service)]


def get_book_upload_service(request: Request, storage: StorageDep) -> BookUploadService:
    return BookUploadService(
        engine=request.app.state.engine,
        storage=storage,
        process_pool=request.app.state.process_pool,
    )


BookUploadServiceDep = Annotated[BookUploadService, Depends(get_book_upload_service)]


def get_book_ingestion_service(
    request: Request, storage: StorageDep
) -> BookIngestionService:
    return BookIngestionService(
        engine=request.app.state.engine,
        storage=storage,
        process_pool=request.app.state.process_pool,
        embedder=ChunkEmbedder(request.app.state.embeddings),
    )


BookIngestionServiceDep = Annotated[
    BookIngestionService, Depends(get_book_ingestion_service)
]
