from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.engine import Engine, make_url
from sqlmodel import Session, create_engine

from ..core.config import DatabaseConfig


def create_db_engine(config: DatabaseConfig) -> Engine:
    database_url = make_url(config.database_url)
    if database_url.drivername in {"postgres", "postgresql"}:
        database_url = database_url.set(drivername="postgresql+psycopg")
    return create_engine(database_url)


def get_session(request: Request) -> Iterator[Session]:
    with Session(request.app.state.engine) as session:
        yield session
