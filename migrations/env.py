"""Alembic uses the application settings and all SQLModel table metadata."""

from alembic import context
from pgvector.sqlalchemy import VECTOR
from sqlmodel import SQLModel

from app.core.config import DatabaseConfig
from app.db.database import create_db_engine
from app.models.book import Book  # noqa: F401 - register table metadata
from app.models.book_chunk import BookChunk  # noqa: F401 - register table metadata
from app.models.book_index import BookIndex  # noqa: F401 - register table metadata

config = context.config
target_metadata = SQLModel.metadata


def render_item(type_, obj, autogen_context):
    if type_ == "type" and isinstance(obj, VECTOR):
        autogen_context.imports.add("from pgvector.sqlalchemy import VECTOR")
        return f"VECTOR({obj.dim})"
    return False


def run_with_connection(connection):
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        render_item=render_item,
        version_table_schema=config.attributes.get("version_table_schema"),
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    # Tests can supply their configured connection without changing environment vars.
    connection = config.attributes.get("connection")
    if connection is not None:
        run_with_connection(connection)
        return
    engine = create_db_engine(DatabaseConfig())
    try:
        with engine.connect() as connection:
            run_with_connection(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    engine = create_db_engine(DatabaseConfig())
    try:
        context.configure(
            url=engine.url,
            target_metadata=target_metadata,
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
        )
        with context.begin_transaction():
            context.run_migrations()
    finally:
        engine.dispose()
else:
    run_migrations_online()
