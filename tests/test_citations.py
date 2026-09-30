import unittest
from uuid import uuid4

from pydantic import ValidationError

from app.models.search import BookSearchResult, RetrievedPassage, SearchHit
from app.schemas.citation import CitationSource
from app.services.ai.citations import CitationRegistry
from app.services.ai.tools.book_search import BookSearch, SearchPhase
from tests.support import FixedEmbeddings


def passage(**changes):
    return RetrievedPassage(
        **{
            "chunk_id": uuid4(),
            "text": "Retrieved evidence",
            "section_path": ["Chapter 3", "Memory"],
            "pdf_pages": [42, 43],
            **changes,
        }
    )


class CitationTests(unittest.TestCase):
    def test_ids_are_unique_across_searches_and_repeated_chunks_are_reused(self):
        registry = CitationRegistry()
        first, second = passage(), passage()
        result = BookSearchResult(book_id=uuid4(), index_id=uuid4(), passages=[first])
        self.assertEqual(registry.register(result).passages[0].citation_id, "s1")
        repeated = registry.register(
            result.model_copy(update={"passages": [second, first]})
        )
        self.assertEqual([p.citation_id for p in repeated.passages], ["s2", "s1"])
        sources = registry.snapshot().sources
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0].pdf_pages, [42, 43])
        self.assertEqual(sources[0].section_path, ["Chapter 3", "Memory"])
        self.assertEqual(sources[0].book_id, result.book_id)
        self.assertEqual(sources[0].index_id, result.index_id)

    def test_new_registry_cannot_reuse_previous_response_sources(self):
        result = BookSearchResult(
            book_id=uuid4(), index_id=uuid4(), passages=[passage()]
        )
        CitationRegistry().register(result)
        self.assertEqual(CitationRegistry().snapshot().sources, [])

    def test_empty_retrieval_has_no_citation_sources(self):
        registry = CitationRegistry()
        result = BookSearchResult(book_id=uuid4(), index_id=uuid4(), passages=[])
        self.assertEqual(registry.register(result).passages, [])
        self.assertEqual(registry.snapshot().sources, [])

    def test_pages_must_be_nonempty_one_based_integers(self):
        for pages in [[], [0], [-1], [True], ["42"], [42.5]]:
            with self.subTest(pages=pages), self.assertRaises(ValidationError):
                passage(pdf_pages=pages)

    def test_source_contract_rejects_invalid_ids_and_extra_fields(self):
        data = {
            "id": "s1",
            "book_id": uuid4(),
            "index_id": uuid4(),
            **passage().model_dump(exclude={"text"}),
        }
        for changes in [{"id": "s0"}, {"id": "s1\n"}, {"url": "invented"}]:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                CitationSource.model_validate({**data, **changes})


class RetrievalCitationTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_converts_storage_pages_once(self):
        class FixedSearch(BookSearch):
            def __init__(self):
                self.book_id = uuid4()
                self.index_id = uuid4()
                self._embeddings = FixedEmbeddings()

            def _search(self, vector):
                return [
                    SearchHit(
                        chunk_id=uuid4(),
                        index_id=self.index_id,
                        text="Evidence",
                        path=("Chapter 1",),
                        pages=(0, 2),
                        distance=0.0,
                    )
                ]

        searcher = FixedSearch()
        updates = [update async for update in searcher.stream_search("question")]
        self.assertEqual(
            updates[:2], [SearchPhase.EMBEDDING_QUERY, SearchPhase.SEARCHING_BOOK]
        )
        result = updates[-1]
        self.assertIsInstance(result, BookSearchResult)
        self.assertEqual(result.passages[0].pdf_pages, [1, 3])
        source = CitationRegistry()
        source.register(result)
        self.assertEqual(source.snapshot().sources[0].pdf_pages, [1, 3])
