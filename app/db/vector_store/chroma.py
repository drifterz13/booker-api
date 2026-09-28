from dataclasses import dataclass
import json
from pathlib import Path
from typing import cast
from uuid import uuid4

import chromadb

from app.models.chunk import Chunk

from ...core.config import Config
from ...models import EmbeddedChunk


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A stored passage returned by vector search."""

    source: Path
    path: tuple[str, ...]
    pages: tuple[int, ...]
    text: str
    distance: float

    @property
    def start_page(self) -> int:
        return self.pages[0] + 1

    @property
    def end_page(self) -> int:
        return self.pages[-1] + 1


class ChromaStore:
    """Keep one chat's book embeddings in memory."""

    def __init__(self, *, config: Config) -> None:
        self._client = chromadb.CloudClient(
            api_key=config.chroma_api_key,
            tenant=config.chroma_tenant,
            database=config.chroma_database,
        )
        self._collection = self._client.create_collection(
            name=f"booker-{uuid4().hex}",
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def close(self) -> None:
        self._client.delete_collection(name=self._collection.name)

    def sync(self, source: Path, items: list[EmbeddedChunk]) -> None:
        """Replace stored items for a source, removing any stale entries."""

        source_id = str(source.resolve())
        old_ids = set(
            self._collection.get(where={"source": source_id}, include=[])["ids"]
        )
        ids = [self._chunk_id(source_id, item.chunk) for item in items]

        for start in range(0, len(items), 100):
            batch = items[start : start + 100]
            self._collection.upsert(
                ids=ids[start : start + 100],
                embeddings=[item.vector for item in batch],
                documents=[item.chunk.text for item in batch],
                metadatas=[
                    {
                        "source": source_id,
                        "path": json.dumps(item.chunk.path, ensure_ascii=False),
                        "pages": json.dumps(item.chunk.pages),
                    }
                    for item in batch
                ],
            )

        if stale_ids := old_ids - set(ids):
            self._collection.delete(ids=list(stale_ids))

    @staticmethod
    def _chunk_id(source_id: str, chunk: Chunk) -> str:
        return f"{source_id}:{chunk.segment_index}:{chunk.chunk_index}"

    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 5,
        source: Path | None = None,
    ) -> list[SearchHit]:
        """Search by a query vector from the same embedding model."""

        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where={"source": str(source.resolve())} if source else None,
            include=["documents", "metadatas", "distances"],
        )

        documents = cast(list[list[str]], result["documents"])
        metadatas = cast(list[list[dict[str, str]]], result["metadatas"])
        distances = cast(list[list[float]], result["distances"])

        return [
            SearchHit(
                source=Path(metadata["source"]),
                path=tuple(json.loads(metadata["path"])),
                pages=tuple(json.loads(metadata["pages"])),
                text=document,
                distance=distance,
            )
            for document, metadata, distance in zip(
                documents[0], metadatas[0], distances[0], strict=True
            )
        ]
