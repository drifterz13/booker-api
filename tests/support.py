"""Shared resources for integration tests against PostgreSQL and RustFS."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import boto3
import pymupdf
from alembic import command
from alembic.config import Config as AlembicConfig
from botocore.client import BaseClient
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from sqlalchemy.engine import Engine

from app.core.config import DatabaseConfig, IntegrationTestConfig, StorageConfig
from app.db.database import create_db_engine


@contextmanager
def database_resources() -> Generator[tuple[DatabaseConfig, Engine], None, None]:
    """Provide a migrated test database and dispose its engine on exit."""
    settings = IntegrationTestConfig()
    database = DatabaseConfig(database_url=settings.test_database_url)
    engine = create_db_engine(database)
    try:
        migrations = AlembicConfig(
            str(Path(__file__).resolve().parents[1] / "alembic.ini")
        )
        with engine.begin() as connection:
            migrations.attributes["connection"] = connection
            command.upgrade(migrations, "head")
        yield database, engine
    finally:
        engine.dispose()


@contextmanager
def storage_resources() -> Generator[tuple[StorageConfig, BaseClient], None, None]:
    """Provide the test bucket and close the S3 client on exit."""
    settings = IntegrationTestConfig()
    storage = StorageConfig(s3_bucket_name=settings.test_s3_bucket_name)
    s3 = boto3.client(
        "s3",
        endpoint_url=storage.s3_endpoint_url,
        aws_access_key_id=storage.aws_access_key_id,
        aws_secret_access_key=storage.aws_secret_key,
        region_name=storage.s3_region_name,
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=3,
            read_timeout=5,
            retries={"total_max_attempts": 1},
        ),
    )
    try:
        try:
            s3.head_bucket(Bucket=storage.s3_bucket_name)
        except ClientError as error:
            if error.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
                raise
            options = {"Bucket": storage.s3_bucket_name}
            if storage.s3_region_name != "us-east-1":
                options["CreateBucketConfiguration"] = {
                    "LocationConstraint": storage.s3_region_name
                }
            s3.create_bucket(**options)
        yield storage, s3
    finally:
        s3.close()


def create_test_pdf() -> bytes:
    """Generate a one-page PDF for upload tests."""
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Book upload integration test")
        return document.tobytes()
