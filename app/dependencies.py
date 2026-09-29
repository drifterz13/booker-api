from typing import Annotated

from fastapi import Depends, Request
from sqlmodel import Session

from .db.database import get_session
from .services.books import BookService
from .services.storage.storage import ObjectStorage

SessionDep = Annotated[Session, Depends(get_session)]


def get_storage(request: Request) -> ObjectStorage:
    return request.app.state.storage


StorageDep = Annotated[ObjectStorage, Depends(get_storage)]


def get_book_service(session: SessionDep, storage: StorageDep) -> BookService:
    return BookService(session=session, storage=storage)


BookServiceDep = Annotated[BookService, Depends(get_book_service)]
