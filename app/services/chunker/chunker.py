from langchain_text_splitters import RecursiveCharacterTextSplitter

from ...models import Chunk, PAGE_SEPARATOR, ContentSegment


class Chunker:
    def __init__(self, *, chunk_size: int = 2000, chunk_overlap: int = 200) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    def chunk(self, segments: list[ContentSegment]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for segment_index, segment in enumerate(segments):
            if not segment.text:
                continue

            # (start, exclusive end, PDF page) in segment.text character offsets.
            page_ranges: list[tuple[int, int, int]] = []
            offset = 0
            for fragment in segment.fragments:
                page_ranges.append((offset, offset + len(fragment.text), fragment.page))
                offset += len(fragment.text) + len(PAGE_SEPARATOR)

            documents = self._splitter.create_documents([segment.text])
            for chunk_index, document in enumerate(documents):
                start = document.metadata["start_index"]
                end = start + len(document.page_content)
                if start < 0 or segment.text[start:end] != document.page_content:
                    raise ValueError(
                        f"Could not locate chunk {chunk_index} in segment "
                        f"{segment_index}"
                    )

                pages = tuple(
                    page
                    for page_start, page_end, page in page_ranges
                    if page_start < end and start < page_end
                )
                if not pages:
                    raise ValueError(
                        f"Chunk {chunk_index} in segment {segment_index} "
                        "has no source page"
                    )

                chunks.append(
                    Chunk(
                        segment_index=segment_index,
                        chunk_index=chunk_index,
                        path=segment.path,
                        level=segment.level,
                        pages=pages,
                        text=document.page_content,
                        word_count=len(document.page_content.split()),
                    )
                )

        return chunks
