import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf

from app.services.ingest.pipeline import chunk_book, extract_book


class OutlineRecoveryTests(unittest.TestCase):
    def test_backward_bookmark_keeps_ingestion_extractable(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "book.pdf"
            with pymupdf.open() as pdf:
                page = pdf.new_page()
                page.insert_text((72, 300), "Text before the backward bookmark")
                page.insert_text((72, 650), "Text after the backward bookmark")
                pdf.set_toc(
                    [
                        [1, "First", 1, 100],
                        [1, "Later", 1, 629],
                        [1, "Backward", 1, 278],
                    ]
                )
                pdf.save(source)

                segments = extract_book(source)

        self.assertEqual([segment.start.y for segment in segments], [100, 629, 629])
        self.assertEqual([segment.end.y for segment in segments[:2]], [629, 629])
        self.assertIn("Text before", segments[0].text)
        self.assertEqual(segments[1].text, "")
        self.assertIn("Text after", segments[2].text)
        self.assertEqual(len(chunk_book(segments)), 2)


if __name__ == "__main__":
    unittest.main()
