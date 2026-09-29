from typing import cast

import pymupdf

from app.models import PdfDocument, PdfPosition
from app.models.content import ContentSegment, PageFragment
from app.services.extractor.outline import PdfExtractionError


class ContentExtractor:
    def extract(self, document: PdfDocument) -> list[ContentSegment]:
        entries = list(document.walk_with_path())
        if not entries:
            return []

        with pymupdf.open(document.source) as pdf:
            document_end = PdfPosition(page=pdf.page_count, y=0.0)
            segments: list[ContentSegment] = []

            for index, (section, path) in enumerate(entries):
                end = (
                    entries[index + 1][0].start
                    if index + 1 < len(entries)
                    else document_end
                )
                fragments = self._extract_fragments(pdf, section.start, end)
                segments.append(
                    ContentSegment(
                        title=section.title,
                        level=section.level,
                        path=path,
                        start=section.start,
                        end=end,
                        fragments=fragments,
                    )
                )

            return segments

    @staticmethod
    def _extract_fragments(
        pdf: pymupdf.Document,
        start: PdfPosition,
        end: PdfPosition,
    ) -> tuple[PageFragment, ...]:
        if end < start:
            raise PdfExtractionError(
                f"Content end {end!r} appears before start {start!r}"
            )

        fragments: list[PageFragment] = []
        last_page = min(end.page, pdf.page_count - 1)

        for page_index in range(start.page, last_page + 1):
            page = pdf[page_index]
            top = start.y if page_index == start.page else page.rect.y0
            bottom = end.y if page_index == end.page else page.rect.y1

            top = min(max(top, page.rect.y0), page.rect.y1)
            bottom = min(max(bottom, page.rect.y0), page.rect.y1)
            if bottom <= top:
                continue

            clip = pymupdf.Rect(page.rect.x0, top, page.rect.x1, bottom)
            text = cast(str, page.get_text("text", clip=clip, sort=True)).strip()
            if text:
                fragments.append(PageFragment(page=page_index, text=text))

        return tuple(fragments)
