import os
from sqlmodel import SQLModel, Session, create_engine

engine = create_engine(
    os.getenv("DATABASE_URL", "postgresql://postgres:admin@mysecret:5432/booker"),
    echo=True,
)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
