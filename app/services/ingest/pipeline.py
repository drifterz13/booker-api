from pathlib import Path

from ...models import Chunk, ContentSegment
from ..chunker import Chunker
from ..extractor import ContentExtractor, OutlineExtractor

PIPELINE_VERSION = "v1"


def extract_book(source: Path) -> list[ContentSegment]:
    """Extract book sections and their PDF text."""
    document = OutlineExtractor(src=source).extract()
    if not document.sections:
        raise ValueError("The PDF has no bookmarks to identify its sections")
    return ContentExtractor().extract(document)


def chunk_book(segments: list[ContentSegment]) -> list[Chunk]:
    """Split extracted sections into chunks with page and section metadata."""
    chunks = Chunker(chunk_size=2000, chunk_overlap=200).chunk(segments)
    if not chunks:
        raise ValueError("The PDF has no extractable section text")
    return chunks
