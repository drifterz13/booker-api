from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from .core.config import DatabaseConfig, StorageConfig
from .db.database import create_db_engine
from .routers.books import router as books_router
from .services.storage.storage import ObjectStorage


def create_app(
    *,
    database_config: DatabaseConfig | None = None,
    storage_config: StorageConfig | None = None,
) -> FastAPI:
    """Build an app with its own database and storage resource lifecycle."""
    database_config = (
        database_config if database_config is not None else DatabaseConfig()
    )
    storage_config = storage_config if storage_config is not None else StorageConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        engine = create_db_engine(database_config)
        try:
            storage = await run_in_threadpool(ObjectStorage, config=storage_config)
            app.state.engine = engine
            app.state.storage = storage
            try:
                yield
            finally:
                await run_in_threadpool(storage.close)
        finally:
            await run_in_threadpool(engine.dispose)

    app = FastAPI(lifespan=lifespan)
    app.include_router(books_router)

    @app.get("/health")
    def check_health() -> str:
        return "ok"

    return app


app = create_app()
