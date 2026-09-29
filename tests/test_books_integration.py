import asyncio
import hashlib
import json
import unittest
from datetime import timedelta
from tempfile import TemporaryFile
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlmodel import Session, select

from app.db.vector_store.pgvector import PgVectorStore
from app.main import create_app
from app.models import BookChunk, BookIndex
from app.models.book import Book
from app.models.book_index import BookIndexStatus
from app.services.books import BookService
from tests.support import (
    create_test_embeddings,
    create_test_pdf,
    database_resources,
    storage_resources,
)


class BooksIntegrationTests(unittest.TestCase):
    """Real API, PostgreSQL, RustFS, and PDF processing; mocked embeddings."""

    @classmethod
    def setUpClass(cls):
        cls.database, cls.engine = cls.enterClassContext(database_resources())
        cls.storage, cls.s3 = cls.enterClassContext(storage_resources())

    def setUp(self):
        self.prefix = f"integration-{uuid4().hex}-"
        self.keys = []
        self.embeddings = create_test_embeddings()
        self.app = create_app(
            database_config=self.database,
            storage_config=self.storage,
            embeddings=self.embeddings,
        )
        self.client = self.enterContext(TestClient(self.app))
        self.addCleanup(self.cleanup_books)
        self.pdf = create_test_pdf()

    def books(self):
        with Session(self.app.state.engine) as session:
            return list(
                session.exec(
                    select(Book).where(Book.filename.startswith(self.prefix))
                ).all()
            )

    def cleanup_books(self):
        with Session(self.app.state.engine) as session:
            for book in session.exec(
                select(Book).where(Book.filename.startswith(self.prefix))
            ).all():
                book.active_index_id = None
                session.add(book)
                session.flush()
                session.exec(delete(BookIndex).where(BookIndex.book_id == book.id))
                session.delete(book)
            session.commit()
        for key in self.keys:
            self.s3.delete_object(Bucket=self.storage.s3_bucket_name, Key=key)

    def store(self, content=None):
        key = f"books/{uuid4()}.pdf"
        self.keys.append(key)
        self.s3.put_object(
            Bucket=self.storage.s3_bucket_name,
            Key=key,
            Body=self.pdf if content is None else content,
            ContentType="application/pdf",
        )
        return key

    def create(self, *, key=None, content=None, suffix="book.pdf", client=None):
        return (client or self.client).post(
            "/books",
            json={
                "filename": self.prefix + suffix,
                "object_key": key or self.store(content),
            },
        )

    def test_create_accepts_stored_object_and_ingests_in_background(self):
        key = self.store()
        response = self.create(key=key, suffix="Book.PDF")
        self.assertEqual(response.status_code, 202, response.text)
        public = response.json()
        self.assertIsNone(public["active_index_id"])
        self.assertEqual(public["ingestion"]["status"], "building")
        # TestClient waits for background tasks, although the HTTP response is sent first.
        detail = self.client.get(f"/books/{public['id']}").json()
        self.assertEqual(detail["ingestion"]["status"], "ready")
        self.assertEqual(detail["active_index_id"], public["ingestion"]["id"])
        book = self.books()[0]
        self.assertEqual(book.object_key, key)
        self.assertEqual(book.checksum, hashlib.sha256(self.pdf).hexdigest())
        self.assertEqual(book.created_at.utcoffset(), timedelta(0))
        with Session(self.app.state.engine) as session:
            index = session.get(BookIndex, book.active_index_id)
            self.assertEqual(index.chunk_count, 1)
            chunk = session.exec(
                select(BookChunk).where(BookChunk.index_id == index.id)
            ).one()
            hits = PgVectorStore(session).search(
                list(chunk.embedding), index_id=index.id
            )
            self.assertIn("Book upload integration test", hits[0].text)
            self.assertEqual(hits[0].pages, (0,))

    def test_duplicate_object_is_rejected_without_another_index(self):
        key = self.store()
        self.assertEqual(self.create(key=key).status_code, 202)
        self.assertEqual(self.create(key=key).status_code, 409)
        self.assertEqual(len(self.books()), 1)
        with Session(self.app.state.engine) as session:
            self.assertEqual(
                len(
                    session.exec(
                        select(BookIndex).where(BookIndex.book_id == self.books()[0].id)
                    ).all()
                ),
                1,
            )
        self.embeddings.aembed_documents.assert_awaited_once()

    def test_response_is_sent_before_embedding_starts(self):
        """Inspect ASGI sends because TestClient waits for background tasks."""
        key = self.store()
        sent = []
        response_finished = False
        original_embed = self.embeddings.aembed_documents.side_effect

        async def embed(texts):
            self.assertTrue(response_finished)
            return original_embed(texts)

        self.embeddings.aembed_documents.side_effect = embed
        payload = json.dumps(
            {"filename": self.prefix + "early.pdf", "object_key": key}
        ).encode()

        async def receive():
            return {"type": "http.request", "body": payload, "more_body": False}

        async def send(message):
            nonlocal response_finished
            sent.append(message)
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                response_finished = True
                self.embeddings.aembed_documents.assert_not_awaited()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/books",
            "raw_path": b"/books",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 1234),
        }
        asyncio.run(self.app(scope, receive, send))
        self.assertEqual(sent[0]["status"], 202)
        public = json.loads(sent[1]["body"])
        self.assertEqual(public["ingestion"]["status"], "building")
        self.embeddings.aembed_documents.assert_awaited_once()
        self.assertEqual(
            self.client.get(f"/books/{public['id']}").json()["ingestion"]["status"],
            "ready",
        )

    def test_list_matches_database_order_and_pagination(self):
        for suffix in ("first.pdf", "second.pdf", "third.pdf"):
            response = self.create(suffix=suffix)
            self.assertEqual(response.status_code, 202, response.text)
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

    def test_invalid_requests_and_missing_objects_do_not_create_books(self):
        key = f"books/{uuid4()}.pdf"
        for payload in (
            {},
            {"filename": "book.txt", "object_key": key},
            {"filename": "book.pdf", "object_key": "other/file.pdf"},
        ):
            self.assertEqual(self.client.post("/books", json=payload).status_code, 422)
        self.assertEqual(self.create(key=key).status_code, 404)
        self.assertEqual(self.client.get(f"/books/{uuid4()}").status_code, 404)
        for params in ({"offset": -1}, {"limit": 0}, {"limit": 101}):
            self.assertEqual(self.client.get("/books", params=params).status_code, 422)
        self.assertEqual(self.books(), [])
        self.embeddings.aembed_documents.assert_not_awaited()

    def test_empty_and_oversized_objects_do_not_create_books(self):
        self.assertEqual(self.create(content=b"").status_code, 400)
        with TemporaryFile() as source:
            source.seek(BookService.MAX_UPLOAD_BYTES)
            source.write(b"\0")
            source.seek(0)
            key = f"books/{uuid4()}.pdf"
            self.keys.append(key)
            self.s3.upload_fileobj(source, self.storage.s3_bucket_name, key)
        self.assertEqual(self.create(key=key).status_code, 413)
        self.assertEqual(self.books(), [])

    def test_storage_failure_does_not_create_book(self):
        missing = self.storage.model_copy(
            update={"s3_bucket_name": f"booker-test-missing-{uuid4().hex}"}
        )
        app = create_app(
            database_config=self.database,
            storage_config=missing,
            embeddings=self.embeddings,
        )
        with TestClient(app) as client:
            response = self.create(key=f"books/{uuid4()}.pdf", client=client)
        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(self.books(), [])

    def test_invalid_content_is_reported_as_failed_after_acceptance(self):
        for content in (b"not a PDF", create_test_pdf(bookmarks=False)):
            with self.subTest(content=content[:20]):
                with self.assertLogs("app.services.ingest.books", level="ERROR"):
                    response = self.create(content=content)
                self.assertEqual(response.status_code, 202, response.text)
                self.assert_failed_ingestion(response)
        self.embeddings.aembed_documents.assert_not_awaited()

    def test_embedding_failure_is_reported_as_failed_after_acceptance(self):
        self.embeddings.aembed_documents.side_effect = RuntimeError(
            "Provider unavailable"
        )
        with self.assertLogs("app.services.ingest.books", level="ERROR"):
            response = self.create()
        self.assertEqual(response.status_code, 202, response.text)
        self.assert_failed_ingestion(response)

    def assert_failed_ingestion(self, response):
        public = response.json()
        detail = self.client.get(f"/books/{public['id']}").json()
        self.assertEqual(detail["status"], "uploaded")
        self.assertIsNone(detail["active_index_id"])
        self.assertEqual(detail["ingestion"]["status"], "failed")
        with Session(self.app.state.engine) as session:
            index = session.get(BookIndex, UUID(public["ingestion"]["id"]))
            self.assertEqual(index.status, BookIndexStatus.FAILED)
            self.assertEqual(
                session.exec(
                    select(BookChunk).where(BookChunk.index_id == index.id)
                ).all(),
                [],
            )
