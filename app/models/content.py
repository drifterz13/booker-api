from dataclasses import dataclass

from .pdf import PdfPosition

PAGE_SEPARATOR = "\n\n"


@dataclass(frozen=True, slots=True)
class PageFragment:
    """Text extracted from one zero-based physical PDF page."""

    page: int
    text: str

    @property
    def page_number(self) -> int:
        return self.page + 1


@dataclass(frozen=True, slots=True)
class ContentSegment:
    """Non-overlapping text owned by one outline entry.

    Example: fragments ``(PageFragment(0, "Hello"),
    PageFragment(1, "World"))`` produce ``text == "Hello\\n\\nWorld"``.
    The page indexes 0 and 1 refer to physical PDF pages 1 and 2.
    """

    title: str
    level: int
    path: tuple[str, ...]
    start: PdfPosition
    end: PdfPosition
    fragments: tuple[PageFragment, ...]

    @property
    def text(self) -> str:
        return PAGE_SEPARATOR.join(fragment.text for fragment in self.fragments)

    @property
    def start_page(self) -> int:
        return self.start.page_number

    @property
    def end_page(self) -> int:
        """Return the last one-based physical page containing segment text."""

        if self.end.page > self.start.page and self.end.y <= 0:
            return self.end.page
        return self.end.page_number
