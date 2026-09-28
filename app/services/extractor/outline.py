from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pymupdf

from app.models import Book, BookSection, Position


class BookExtractionError(ValueError):
    """Raised when a PDF outline cannot be represented safely."""


class OutlineExtractor:
    def __init__(self, *, src: Path) -> None:
        self._src = src

    def extract(self) -> Book:

        with pymupdf.open(self._src) as pdf:
            roots = self._build_sections(pdf)
            return Book(
                source=self._src,
                page_count=pdf.page_count,
                sections=roots,
            )

    def _build_sections(
        self,
        pdf: pymupdf.Document,
    ) -> list[BookSection]:
        toc = pdf.get_toc(simple=False)
        roots: list[BookSection] = []
        open_sections: list[BookSection] = []

        for toc_index, item in enumerate(toc):
            level, title, page_number, destination = item

            if toc_index == 0 and level != 1:
                raise BookExtractionError("The first outline entry must be level 1")

            start = self._bookmark_position(
                pdf,
                title=title,
                page_number=page_number,
                destination=destination,
            )

            # A section ends at the next entry on the same or a shallower level.
            while open_sections and open_sections[-1].level >= level:
                open_sections.pop().end = start

            if open_sections and level > open_sections[-1].level + 1:
                raise BookExtractionError(
                    f"Outline jumps from level {open_sections[-1].level} "
                    f"to level {level} at entry {toc_index}: {title!r}"
                )

            section = BookSection(
                title=title.strip(),
                level=level,
                start=start,
            )

            if open_sections:
                open_sections[-1].children.append(section)
            else:
                roots.append(section)

            open_sections.append(section)

        document_end = Position(page=pdf.page_count, y=0.0)
        for section in open_sections:
            section.end = document_end

        return roots

    @staticmethod
    def _bookmark_position(
        pdf: pymupdf.Document,
        *,
        title: str,
        page_number: int,
        destination: dict[str, Any],
    ) -> Position:
        page_index = page_number - 1

        if not 0 <= page_index < pdf.page_count:
            raise BookExtractionError(
                f"Outline entry {title!r} has no valid internal page destination"
            )

        page = pdf[page_index]
        point = destination.get("to")
        y = getattr(point, "y", page.rect.y0)

        if not isinstance(y, (int, float)) or not math.isfinite(y):
            y = page.rect.y0

        # Real PDFs commonly contain bookmark positions just outside the crop box.
        y = max(page.rect.y0, min(float(y), page.rect.y1))
        return Position(page=page_index, y=y)
