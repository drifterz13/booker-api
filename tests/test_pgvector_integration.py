import unittest
from uuid import uuid4

from sqlalchemy import delete
from sqlmodel import Session

from app.db.vector_store.pgvector import PgVectorStore
from app.models import Book, BookIndex, Chunk, EmbeddedChunk
from app.models.book_index import EMBEDDING_DIMENSIONS
from tests.support import database_resources


def embedded(position, x, y):
    return EmbeddedChunk(
        chunk=Chunk(
            segment_index=0,
            chunk_index=position,
            path=("Chapter 1",),
            level=1,
            pages=(0, 1),
            text=f"Chunk {position}",
            word_count=2,
        ),
        vector=[x, y] + [0.0] * (EMBEDDING_DIMENSIONS - 2),
    )


class PgVectorIntegrationTests(unittest.TestCase):
    """Save and search chunks using the real booker-test database."""

    @classmethod
    def setUpClass(cls):
        cls.database, cls.engine = cls.enterClassContext(database_resources())

    def cleanup_book(self, book_id):
        with Session(self.engine) as session:
            # Deleting index versions also deletes their chunks.
            session.exec(delete(BookIndex).where(BookIndex.book_id == book_id))
            session.exec(delete(Book).where(Book.id == book_id))
            session.commit()

    def test_save_and_search_chunks(self):
        book = Book(
            filename="vector-test.pdf",
            object_key=f"books/{uuid4()}.pdf",
            checksum="a" * 64,
        )
        self.addCleanup(self.cleanup_book, book.id)
        with Session(self.engine) as session:
            session.add(book)
            session.flush()
            index = BookIndex(book_id=book.id, pipeline_version="test-v1")
            other = BookIndex(book_id=book.id, pipeline_version="test-v2")
            session.add_all([index, other])
            session.flush()
            index_id = index.id
            store = PgVectorStore(session)
            self.assertEqual(
                store.add_chunks(
                    index_id, [embedded(0, 1, 0), embedded(1, 0, 1), embedded(2, -1, 0)]
                ),
                3,
            )
            store.add_chunks(other.id, [embedded(99, 1, 0)])
            session.commit()

        # A new session proves the chunks were persisted.
        with Session(self.engine) as session:
            hits = PgVectorStore(session).search(
                embedded(0, 1, 0).vector, index_id=index_id, limit=2
            )
            self.assertEqual([hit.text for hit in hits], ["Chunk 0", "Chunk 1"])
            self.assertTrue(all(hit.index_id == index_id for hit in hits))
            self.assertAlmostEqual(hits[0].distance, 0.0)
            self.assertAlmostEqual(hits[1].distance, 1.0)
            self.assertEqual(hits[0].path, ("Chapter 1",))
            self.assertEqual(hits[0].pages, (0, 1))


if __name__ == "__main__":
    unittest.main()
