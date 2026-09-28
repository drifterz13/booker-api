from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrieval-sized piece of one content segment.

    For a chunk containing ``"Hello\\n\\nWorld"`` from two PDF pages,
    ``pages == (0, 1)``, ``start_page == 1``, and ``end_page == 2``.
    A chunk contained on the first page has ``pages == (0,)``.
    """

    segment_index: int
    chunk_index: int
    path: tuple[str, ...]
    level: int
    pages: tuple[int, ...]
    text: str
    word_count: int

    @property
    def embedding_text(self) -> str:
        context = " > ".join(self.path)
        return f"{context}\n\n{self.text}"

    @property
    def start_page(self) -> int:
        return self.pages[0] + 1

    @property
    def end_page(self) -> int:
        return self.pages[-1] + 1
