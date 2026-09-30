import multiprocessing
import unittest
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from uuid import uuid4

import pymupdf
from sqlalchemy import delete
from sqlmodel import Session, select

from app.db.vector_store.pgvector import PgVectorStore
from app.models import Book, BookChunk, BookIndex
from app.models.book_index import BookIndexStatus
from app.services.chunker import ChunkEmbedder
from app.services.ingest.books import BookIngestionError, BookIngestionService
from app.services.storage.storage import ObjectStorage
from tests.support import create_test_embeddings, database_resources, storage_resources


class IngestionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Real extraction and storage with fixed embeddings instead of OpenAI calls."""

    @classmethod
    def setUpClass(cls):
        cls.database, cls.engine = cls.enterClassContext(database_resources())
        cls.storage_config, cls.s3 = cls.enterClassContext(storage_resources())
        cls.storage = ObjectStorage(config=cls.storage_config)
        cls.addClassCleanup(cls.storage.close)
        cls.pool = cls.enterClassContext(
            ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn")
            )
        )

    def cleanup_book(self, book_id, object_key):
        self.s3.delete_object(Bucket=self.storage_config.s3_bucket_name, Key=object_key)
        self.s3.delete_object(
            Bucket=self.storage_config.s3_bucket_name,
            Key=f"thumbnails/{book_id}.png",
        )
        with Session(self.engine) as session:
            book = session.get(Book, book_id)
            if book is not None:
                book.active_index_id = None
                session.add(book)
                session.flush()
                session.exec(delete(BookIndex).where(BookIndex.book_id == book_id))
                session.delete(book)
                session.commit()

    async def test_ingest_and_failed_retry_preserves_active_index(self):
        book = Book(
            filename="ingestion-test.pdf",
            object_key=f"books/{uuid4()}.pdf",
            checksum="a" * 64,
        )
        book_id, object_key = book.id, book.object_key
        self.addCleanup(self.cleanup_book, book_id, object_key)
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text(
                (72, 100), "PostgreSQL stores book chunks and their embeddings."
            )
            content = pdf.tobytes()
        with pymupdf.open() as pdf:
            pdf.new_page()
            without_text = pdf.tobytes()
        self.storage.upload(BytesIO(content), object_key=object_key)
        with Session(self.engine) as session:
            session.add(book)
            session.commit()

        service = BookIngestionService(
            engine=self.engine,
            storage=self.storage,
            process_pool=self.pool,
            embedder=ChunkEmbedder(create_test_embeddings()),
        )
        index_id = await service.ingest(book_id)
        with Session(self.engine) as session:
            index = session.get(BookIndex, index_id)
            self.assertEqual(index.status, BookIndexStatus.READY)
            self.assertEqual(index.chunk_count, 1)
            self.assertEqual(session.get(Book, book_id).active_index_id, index_id)
            chunk = session.exec(
                select(BookChunk).where(BookChunk.index_id == index_id)
            ).one()
            hits = PgVectorStore(session).search(
                list(chunk.embedding), index_id=index_id
            )
            self.assertIn("PostgreSQL", hits[0].text)
            self.assertEqual(hits[0].path, ())
            self.assertEqual(hits[0].pages, (0,))

        # A real extraction failure must preserve the ready version.
        self.storage.upload(BytesIO(without_text), object_key=object_key)
        with self.assertRaises(BookIngestionError) as failed:
            await service.ingest(book_id)
        with Session(self.engine) as session:
            self.assertEqual(session.get(Book, book_id).active_index_id, index_id)
            self.assertEqual(
                session.get(BookIndex, failed.exception.index_id).status,
                BookIndexStatus.FAILED,
            )
            self.assertEqual(
                session.exec(
                    select(BookChunk).where(
                        BookChunk.index_id == failed.exception.index_id
                    )
                ).all(),
                [],
            )


if __name__ == "__main__":
    unittest.main()
