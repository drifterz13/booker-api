from typing import Annotated
from uuid import UUID

from pydantic import Field, FiniteFloat, TypeAdapter
from sqlalchemy import text
from sqlmodel import Session, select

from ...models import BookChunk, BookIndex, EmbeddedChunk
from ...models.book_index import EMBEDDING_DIMENSIONS, BookIndexStatus
from ...models.search import SearchHit

Embedding = Annotated[
    list[FiniteFloat],
    Field(min_length=EMBEDDING_DIMENSIONS, max_length=EMBEDDING_DIMENSIONS),
]
_embedding = TypeAdapter(Embedding)


def _validate_vector(values: list[float]) -> list[float]:
    vector = _embedding.validate_python(values)
    if not any(vector):
        raise ValueError("Cosine search requires a nonzero embedding")
    return vector


class PgVectorStore:
    """Store and search embedded chunks for a book index.

    The caller owns the session, commit/rollback, and index activation. Writes
    lock the index row until the transaction ends; lifecycle changes must use
    the same row lock. Ready and failed indexes cannot receive more chunks.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_chunks(self, index_id: UUID, items: list[EmbeddedChunk]) -> int:
        """Add embedded chunks to a building index without committing."""
        if not items:
            raise ValueError("At least one embedded chunk is required")
        chunks = []
        positions = set()
        for item in items:
            chunk = BookChunk.model_validate(
                {
                    "index_id": index_id,
                    "segment_index": item.chunk.segment_index,
                    "chunk_index": item.chunk.chunk_index,
                    "content": item.chunk.text,
                    "section_path": list(item.chunk.path),
                    "pdf_pages": list(item.chunk.pages),
                    "embedding": _validate_vector(item.vector),
                }
            )
            if any(page < 0 for page in chunk.pdf_pages):
                raise ValueError("PDF page indices must be non-negative")
            position = (chunk.segment_index, chunk.chunk_index)
            if position in positions:
                raise ValueError("Duplicate chunk position in batch")
            positions.add(position)
            chunks.append(chunk)

        index = self._session.get(
            BookIndex, index_id, with_for_update=True, populate_existing=True
        )
        if index is None:
            raise ValueError("Book index does not exist")
        if index.status != BookIndexStatus.BUILDING:
            raise ValueError("Only building indexes can receive chunks")

        self._session.add_all(chunks)
        self._session.flush()
        return len(chunks)

    def search(
        self,
        query_vector: list[float],
        *,
        index_id: UUID,
        limit: int = 5,
    ) -> list[SearchHit]:
        """Return the closest chunks from the specified book index.

        The caller resolves the selected ready index. PostgreSQL chooses an
        exact or HNSW scan; iterative HNSW scans improve filtered retrieval.
        """
        if limit < 1:
            raise ValueError("Search limit must be positive")
        vector = _validate_vector(query_vector)
        self._session.exec(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
        distance = BookChunk.embedding.cosine_distance(vector)
        statement = (
            select(
                BookChunk.id,
                BookChunk.content,
                BookChunk.section_path,
                BookChunk.pdf_pages,
                distance.label("distance"),
            )
            .where(BookChunk.index_id == index_id)
            .order_by(distance)
            .limit(limit)
        )
        return [
            SearchHit(
                chunk_id=chunk_id,
                index_id=index_id,
                text=content,
                path=tuple(section_path),
                pages=tuple(pdf_pages),
                distance=float(score),
            )
            for chunk_id, content, section_path, pdf_pages, score in self._session.exec(
                statement
            )
        ]
