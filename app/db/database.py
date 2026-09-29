from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.engine import Engine, make_url
from sqlmodel import Session, SQLModel, create_engine

from ..core.config import DatabaseConfig
from ..models.book import Book  # noqa: F401 - register the table before create_all


def create_db_engine(config: DatabaseConfig) -> Engine:
    database_url = make_url(config.database_url)
    if database_url.drivername in {"postgres", "postgresql"}:
        database_url = database_url.set(drivername="postgresql+psycopg")
    return create_engine(database_url)


def create_db_and_tables(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)


def get_session(request: Request) -> Iterator[Session]:
    with Session(request.app.state.engine) as session:
        yield session
