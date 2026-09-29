import hashlib
import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryFile
from uuid import uuid4

import boto3
import pymupdf
from alembic import command
from alembic.config import Config as AlembicConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlmodel import Session, select

from app.core.config import DatabaseConfig, IntegrationTestConfig, StorageConfig
from app.db.database import create_db_engine
from app.main import create_app
from app.models.book import Book, BookStatus
from app.services.books import BookService


class BooksIntegrationTests(unittest.TestCase):
    """Real API, PostgreSQL, and RustFS; no mocks or SQLite replacements."""

    @classmethod
    def setUpClass(cls):
        config = IntegrationTestConfig()
        cls.database = DatabaseConfig(database_url=config.test_database_url)
        cls.storage = StorageConfig(s3_bucket_name=config.test_s3_bucket_name)
        engine = create_db_engine(cls.database)
        try:
            with engine.begin() as connection:
                migrations = AlembicConfig(
                    str(Path(__file__).resolve().parents[1] / "alembic.ini")
                )
                migrations.attributes["connection"] = connection
                command.upgrade(migrations, "head")
        finally:
            engine.dispose()

        cls.s3 = boto3.client(
            "s3",
            endpoint_url=cls.storage.s3_endpoint_url,
            aws_access_key_id=cls.storage.aws_access_key_id,
            aws_secret_access_key=cls.storage.aws_secret_key,
            region_name=cls.storage.s3_region_name,
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                connect_timeout=3,
                read_timeout=5,
                retries={"total_max_attempts": 1},
            ),
        )
        cls.addClassCleanup(cls.s3.close)
        try:
            cls.s3.head_bucket(Bucket=cls.storage.s3_bucket_name)
        except ClientError as error:
            if error.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
                raise
            options = {"Bucket": cls.storage.s3_bucket_name}
            if cls.storage.s3_region_name != "us-east-1":
                options["CreateBucketConfiguration"] = {
                    "LocationConstraint": cls.storage.s3_region_name
                }
            cls.s3.create_bucket(**options)

    def setUp(self):
        self.prefix = f"integration-{uuid4().hex}-"
        self.app = create_app(
            database_config=self.database, storage_config=self.storage
        )
        self.client = self.enterContext(TestClient(self.app))
        # Registered after the client so cleanup runs before its engine closes.
        self.addCleanup(self.cleanup_books)
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text((72, 72), "Book upload integration test")
            self.pdf = document.tobytes()

    def books(self):
        with Session(self.app.state.engine) as session:
            return list(
                session.exec(
                    select(Book).where(Book.filename.startswith(self.prefix))
                ).all()
            )

    def cleanup_books(self):
        with Session(self.app.state.engine) as session:
            books = session.exec(
                select(Book).where(Book.filename.startswith(self.prefix))
            ).all()
            for book in books:
                self.s3.delete_object(
                    Bucket=self.storage.s3_bucket_name, Key=book.object_key
                )
                session.delete(book)
            session.commit()
        if self.books():
            raise AssertionError("Test book records were not cleaned up")
        for book in books:
            try:
                self.s3.head_object(
                    Bucket=self.storage.s3_bucket_name, Key=book.object_key
                )
            except ClientError as error:
                if error.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                    continue
                raise
            raise AssertionError(f"Test object was not cleaned up: {book.object_key}")

    def upload(self, *, suffix="book.pdf", content=None, client=None):
        return (client or self.client).post(
            "/books",
            files={
                "file": (
                    self.prefix + suffix,
                    self.pdf if content is None else content,
                    "application/pdf",
                )
            },
        )

    def object_keys(self):
        return {
            item["Key"]
            for page in self.s3.get_paginator("list_objects_v2").paginate(
                Bucket=self.storage.s3_bucket_name
            )
            for item in page.get("Contents", [])
        }

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "ok")

    def test_pgvector_extension_and_cosine_distance(self):
        with Session(self.app.state.engine) as session:
            version = session.exec(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).one()[0]
            identical, orthogonal = session.exec(
                text(
                    "SELECT '[1,0,0]'::vector <=> '[1,0,0]'::vector, "
                    "'[1,0,0]'::vector <=> '[0,1,0]'::vector"
                )
            ).one()
        self.assertGreaterEqual(tuple(map(int, version.split("."))), (0, 8, 0))
        self.assertAlmostEqual(identical, 0.0)
        self.assertAlmostEqual(orthogonal, 1.0)

    def test_upload_persists_metadata_and_matching_pdf_bytes(self):
        response = self.upload(suffix="Book.PDF")
        self.assertEqual(response.status_code, 201, response.text)
        public = response.json()
        self.assertEqual(set(public), {"id", "filename", "status", "created_at"})
        self.assertEqual(public["status"], "uploaded")
        book = self.books()[0]
        self.assertEqual(str(book.id), public["id"])
        self.assertEqual(book.checksum, hashlib.sha256(self.pdf).hexdigest())
        self.assertEqual(book.status, BookStatus.UPLOADED)
        self.assertEqual(book.created_at.utcoffset(), timedelta(0))
        self.assertEqual(book.object_key, f"books/{book.id}.pdf")
        stored = self.s3.get_object(
            Bucket=self.storage.s3_bucket_name, Key=book.object_key
        )
        self.assertEqual(stored["ContentType"], "application/pdf")
        try:
            self.assertEqual(stored["Body"].read(), self.pdf)
        finally:
            stored["Body"].close()

    def test_same_filename_uploads_have_distinct_objects(self):
        for _ in range(2):
            response = self.upload()
            self.assertEqual(response.status_code, 201, response.text)
        books = self.books()
        self.assertEqual(len(books), 2)
        self.assertEqual(len({book.object_key for book in books}), 2)
        self.assertEqual(len({book.checksum for book in books}), 1)
        self.assertTrue({book.object_key for book in books} <= self.object_keys())

    def test_list_matches_database_order_and_pagination(self):
        for suffix in ("first.pdf", "second.pdf", "third.pdf"):
            response = self.upload(suffix=suffix)
            self.assertEqual(response.status_code, 201, response.text)
        with Session(self.app.state.engine) as session:
            expected = session.exec(
                select(Book)
                .order_by(Book.created_at.desc(), Book.id.desc())
                .offset(1)
                .limit(2)
            ).all()
            expected_ids = [str(book.id) for book in expected]
        response = self.client.get("/books", params={"offset": 1, "limit": 2})
        self.assertEqual(response.status_code, 200)
        self.assertEqual([book["id"] for book in response.json()], expected_ids)
        for book in response.json():
            self.assertEqual(set(book), {"id", "filename", "status", "created_at"})

    def test_invalid_request_fields_are_rejected_without_writes(self):
        before = self.object_keys()
        self.assertEqual(self.client.post("/books").status_code, 422)
        self.assertEqual(self.upload(suffix="book.txt").status_code, 422)
        for params in ({"offset": -1}, {"limit": 0}, {"limit": 101}, {"limit": "abc"}):
            with self.subTest(params=params):
                self.assertEqual(
                    self.client.get("/books", params=params).status_code, 422
                )
        self.assertEqual(self.books(), [])
        self.assertEqual(self.object_keys(), before)

    def test_empty_or_invalid_pdf_is_rejected_without_writes(self):
        before = self.object_keys()
        for content in (b"", b"not a PDF", b"%PDF-1.7\nnot a readable PDF"):
            with self.subTest(content=content):
                self.assertEqual(self.upload(content=content).status_code, 400)
        self.assertEqual(self.books(), [])
        self.assertEqual(self.object_keys(), before)

    def test_actual_upload_size_limit_is_enforced(self):
        before = self.object_keys()
        with TemporaryFile() as source:
            source.write(self.pdf)
            source.seek(BookService.MAX_UPLOAD_BYTES)
            source.write(b"\0")
            source.seek(0)
            response = self.upload(content=source)
        self.assertEqual(response.status_code, 413, response.text)
        self.assertEqual(self.books(), [])
        self.assertEqual(self.object_keys(), before)

    def test_missing_bucket_returns_502_and_persists_failed_book(self):
        missing = self.storage.model_copy(
            update={"s3_bucket_name": f"booker-test-missing-{uuid4().hex}"}
        )
        failed_app = create_app(database_config=self.database, storage_config=missing)
        with TestClient(failed_app) as client:
            response = self.upload(client=client)
        self.assertEqual(response.status_code, 502, response.text)
        book = self.books()[0]
        self.assertEqual(book.status, BookStatus.FAILED)
        self.assertEqual(response.json()["detail"]["book_id"], str(book.id))
        self.assertNotIn(book.object_key, self.object_keys())
        listing = self.client.get("/books", params={"limit": 100}).json()
        self.assertIn(str(book.id), [item["id"] for item in listing])


if __name__ == "__main__":
    unittest.main()
