from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, order=True, slots=True)
class Position:
    """A position in a PDF, using a zero-based physical page index."""

    page: int
    y: float = 0.0

    @property
    def page_number(self) -> int:
        """Return the one-based physical page number used for display."""

        return self.page + 1


@dataclass(slots=True)
class BookSection:
    """One entry in a book's outline.

    ``start`` is inclusive and ``end`` is exclusive. The range covers the
    complete subtree, including all descendant sections.
    """

    title: str
    level: int
    start: Position
    end: Position | None = None
    children: list[BookSection] = field(default_factory=list)

    @property
    def start_page(self) -> int:
        """Return the one-based physical page on which this section starts."""

        return self.start.page_number

    @property
    def end_page(self) -> int | None:
        """Return the last one-based physical page containing section text."""

        if self.end is None:
            return None

        # An end at the top of a later page means that page is not included.
        if self.end.page > self.start.page and self.end.y <= 0:
            return self.end.page

        return self.end.page_number


@dataclass(slots=True)
class Book:
    source: Path
    page_count: int
    sections: list[BookSection] = field(default_factory=list)

    def walk(self) -> Iterator[BookSection]:
        """Yield every outline section in document order."""

        for section, _path in self.walk_with_path():
            yield section

    def walk_with_path(
        self,
    ) -> Iterator[tuple[BookSection, tuple[str, ...]]]:
        """Yield every section together with its title path."""

        stack: list[tuple[BookSection, tuple[str, ...]]] = [
            (section, (section.title,)) for section in reversed(self.sections)
        ]
        while stack:
            section, path = stack.pop()
            yield section, path
            stack.extend(
                (child, (*path, child.title)) for child in reversed(section.children)
            )

    @property
    def max_level(self) -> int:
        return max((section.level for section in self.walk()), default=0)
