from pathlib import Path

from langchain_core.embeddings import Embeddings

from ...db.vector_store import ChromaStore
from ...services.chunker import ChunkEmbedder, Chunker
from ...services.extractor import ContentExtractor, OutlineExtractor


def reindex(
    source: Path,
    *,
    store: ChromaStore,
    embeddings: Embeddings,
) -> int:
    document = OutlineExtractor(src=source).extract()
    if not document.sections:
        raise ValueError("The PDF has no bookmarks to identify its sections")

    segments = ContentExtractor().extract(document)
    chunks = Chunker(chunk_size=2000, chunk_overlap=200).chunk(segments)
    if not chunks:
        raise ValueError("The PDF has no extractable section text")

    embedded = ChunkEmbedder(embeddings).embed(chunks)
    store.sync(source, embedded)
    return len(chunks)
