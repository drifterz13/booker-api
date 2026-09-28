from pathlib import Path

from langchain_core.embeddings import Embeddings

from ...services.chunker import Chunker, ChunkEmbedder
from ...services.extractor import OutlineExtractor, ContentExtractor
from ...db.vector_store import ChromaStore


def reindex(
    source: Path,
    *,
    store: ChromaStore,
    embeddings: Embeddings,
) -> int:
    book = OutlineExtractor(src=source).extract()
    if not book.sections:
        raise ValueError("The PDF has no bookmarks to identify its sections")

    segments = ContentExtractor().extract(book)
    chunks = Chunker(chunk_size=2000, chunk_overlap=200).chunk(segments)
    if not chunks:
        raise ValueError("The PDF has no extractable section text")

    embedded = ChunkEmbedder(embeddings).embed(chunks)
    store.sync(source, embedded)
    return len(chunks)
