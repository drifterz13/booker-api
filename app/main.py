import multiprocessing
from collections.abc import AsyncGenerator
from concurrent.futures import ProcessPoolExecutor
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from .core.config import (
    CorsConfig,
    DatabaseConfig,
    EmbeddingConfig,
    IngestionConfig,
    StorageConfig,
)
from .db.database import create_db_engine
from .models.book_index import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from .routers.books import router as books_router
from .routers.uploads import router as uploads_router
from .services.storage.storage import ObjectStorage


def create_app(
    *,
    database_config: DatabaseConfig | None = None,
    storage_config: StorageConfig | None = None,
    ingestion_config: IngestionConfig | None = None,
    cors_config: CorsConfig | None = None,
    embeddings: Embeddings | None = None,
) -> FastAPI:
    """Build an app with database, storage, process-pool, and client lifecycles."""
    database_config = (
        database_config if database_config is not None else DatabaseConfig()
    )
    storage_config = storage_config if storage_config is not None else StorageConfig()
    ingestion_config = (
        ingestion_config if ingestion_config is not None else IngestionConfig()
    )
    cors_config = cors_config if cors_config is not None else CorsConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        async with AsyncExitStack() as resources:
            engine = create_db_engine(database_config)
            resources.push_async_callback(run_in_threadpool, engine.dispose)
            storage = await run_in_threadpool(ObjectStorage, config=storage_config)
            resources.push_async_callback(run_in_threadpool, storage.close)
            pool = ProcessPoolExecutor(
                max_workers=ingestion_config.ingestion_workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
            resources.push_async_callback(run_in_threadpool, pool.shutdown)
            provider = embeddings
            if provider is None:
                config = EmbeddingConfig()
                sync_client = httpx.Client()
                resources.push_async_callback(run_in_threadpool, sync_client.close)
                async_client = await resources.enter_async_context(httpx.AsyncClient())
                provider = OpenAIEmbeddings(
                    model=EMBEDDING_MODEL,
                    dimensions=EMBEDDING_DIMENSIONS,
                    api_key=config.openai_api_key,
                    http_client=sync_client,
                    http_async_client=async_client,
                    request_timeout=30,
                    max_retries=2,
                )
            app.state.engine = engine
            app.state.storage = storage
            app.state.process_pool = pool
            app.state.embeddings = provider
            yield

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.cors_allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    app.include_router(books_router)
    app.include_router(uploads_router)

    @app.get("/health")
    def check_health() -> str:
        return "ok"

    return app


app = create_app()
