import hashlib
import unittest
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlmodel import Session, select

from app.db.vector_store.pgvector import PgVectorStore
from app.main import create_app
from app.models import BookChunk, BookIndex
from app.models.book import Book
from tests.support import (
    create_test_embeddings,
    create_test_pdf,
    database_resources,
    storage_resources,
)


class BooksIntegrationTests(unittest.TestCase):
    """Real API, PostgreSQL, RustFS, and PDFs with fixed local embeddings."""

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
                self.s3.delete_object(
                    Bucket=self.storage.s3_bucket_name,
                    Key=f"thumbnails/{book.id}.png",
                )
                book.active_index_id = None
                session.add(book)
                session.flush()
                session.exec(delete(BookIndex).where(BookIndex.book_id == book.id))
                session.delete(book)
            session.commit()
        for key in self.keys:
            self.s3.delete_object(Bucket=self.storage.s3_bucket_name, Key=key)

    def store(self):
        key = f"books/{uuid4()}.pdf"
        self.keys.append(key)
        self.s3.put_object(
            Bucket=self.storage.s3_bucket_name,
            Key=key,
            Body=self.pdf,
            ContentType="application/pdf",
        )
        return key

    def create(self, *, key=None, suffix="book.pdf"):
        return self.client.post(
            "/books",
            json={
                "filename": self.prefix + suffix,
                "object_key": key or self.store(),
            },
        )

    def test_create_accepts_stored_object_and_ingests_in_background(self):
        key = self.store()
        response = self.create(key=key, suffix="Book.PDF")
        self.assertEqual(response.status_code, 202, response.text)
        public = response.json()
        self.assertIsNone(public["active_index_id"])
        self.assertIsNone(public["thumbnail_url"])
        self.assertEqual(public["ingestion"]["status"], "building")
        # TestClient waits for background tasks, although the HTTP response is sent first.
        detail = self.client.get(f"/books/{public['id']}").json()
        self.assertEqual(detail["ingestion"]["status"], "ready")
        self.assertEqual(detail["active_index_id"], public["ingestion"]["id"])
        book = self.books()[0]
        self.assertEqual(book.object_key, key)
        self.assertEqual(book.checksum, hashlib.sha256(self.pdf).hexdigest())
        self.assert_thumbnail(public["id"])
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

    def test_pdf_url_supports_browser_downloads(self):
        key = self.store()
        book = Book(filename=self.prefix + "viewer.pdf", object_key=key)
        book_id = book.id
        with Session(self.app.state.engine) as session:
            session.add(book)
            session.commit()
        origin = "http://localhost:5173"
        response = self.client.get(f"/books/{book_id}/pdf", headers={"Origin": origin})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["access-control-allow-origin"], "*")
        url = response.json()["url"]
        with httpx.Client(timeout=10, trust_env=False) as client:
            full = client.get(url, headers={"Origin": origin})
        self.assertEqual(full.status_code, 200)
        self.assertEqual(full.content, self.pdf)
        self.assertEqual(full.headers["content-type"], "application/pdf")
        self.assertIn(full.headers["access-control-allow-origin"], ("*", origin))

    def assert_thumbnail(self, book_id):
        detail = self.client.get(f"/books/{book_id}").json()
        self.assertIsNotNone(detail["thumbnail_url"])
        with httpx.Client(timeout=10, trust_env=False) as client:
            response = client.get(detail["thumbnail_url"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertTrue(response.content.startswith(b"\x89PNG\r\n\x1a\n"))
