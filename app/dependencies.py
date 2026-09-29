from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request
from openai import AsyncOpenAI
from pydantic import ValidationError
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from sqlmodel import Session

from .core.config import ChatConfig
from .db.database import get_session
from .services.ai.tools.book_search import (
    BookNotFoundError,
    BookNotReadyError,
    BookSearch,
)
from .services.books import BookService
from .services.chunker import ChunkEmbedder
from .services.ingest.books import BookIngestionService
from .services.storage.storage import ObjectStorage

SessionDep = Annotated[Session, Depends(get_session)]


def get_storage(request: Request) -> ObjectStorage:
    return request.app.state.storage


StorageDep = Annotated[ObjectStorage, Depends(get_storage)]


def get_book_service(session: SessionDep, storage: StorageDep) -> BookService:
    return BookService(session=session, storage=storage)


BookServiceDep = Annotated[BookService, Depends(get_book_service)]


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


def get_book_search(book_id: UUID, request: Request) -> BookSearch:
    try:
        return BookSearch(
            engine=request.app.state.engine,
            embeddings=request.app.state.embeddings,
            book_id=book_id,
        )
    except BookNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except BookNotReadyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


BookSearchDep = Annotated[BookSearch, Depends(get_book_search)]


def get_chat_config() -> ChatConfig:
    try:
        return ChatConfig()
    except ValidationError as error:
        raise HTTPException(status_code=503, detail="Chat is not configured") from error


ChatConfigDep = Annotated[ChatConfig, Depends(get_chat_config)]


async def get_chat_model(config: ChatConfigDep) -> AsyncIterator[Model]:
    async with AsyncOpenAI(
        api_key=config.openai_api_key.get_secret_value(),
    ) as client:
        yield OpenAIChatModel(
            config.chat_model_name, provider=OpenAIProvider(openai_client=client)
        )


ChatModelDep = Annotated[Model, Depends(get_chat_model)]
