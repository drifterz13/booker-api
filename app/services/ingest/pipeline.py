from pathlib import Path

import pymupdf

from ...models import Chunk, ContentSegment
from ..books import BookValidationError
from ..chunker import Chunker
from ..extractor import ContentExtractor, OutlineExtractor

PIPELINE_VERSION = "v1"


class BookContentError(ValueError):
    """A readable PDF cannot be processed by the book pipeline."""


def validate_pdf(content: bytes) -> None:
    """Check PDF readability in the worker process before storing an upload."""
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if not document.is_pdf or document.page_count == 0:
                raise BookValidationError("The upload must be a PDF with pages")
    except pymupdf.FileDataError as error:
        raise BookValidationError("The upload is not a readable PDF") from error


def extract_book(source: Path) -> list[ContentSegment]:
    """Extract book sections and their PDF text."""
    document = OutlineExtractor(src=source).extract()
    if not document.sections:
        raise BookContentError("The PDF has no bookmarks to identify its sections")
    return ContentExtractor().extract(document)


def chunk_book(segments: list[ContentSegment]) -> list[Chunk]:
    """Split extracted sections into chunks with page and section metadata."""
    chunks = Chunker(chunk_size=2000, chunk_overlap=200).chunk(segments)
    if not chunks:
        raise BookContentError("The PDF has no extractable section text")
    return chunks
