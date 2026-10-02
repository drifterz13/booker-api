import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from pydantic_ai.models import Model
from pydantic_ai.run import AgentRunResultEvent
from pydantic_core import to_json

from app.models.search import BookSearchResult
from app.services.ai.chat import BookChatDeps, create_book_agent
from app.services.ai.tools.book_search import BookSearch, SearchPhase


def write_artifact(path: Path, artifact: dict) -> None:
    # Pydantic usage costs contain Decimal values; preserve their precision
    # as JSON strings rather than converting them to floats.
    path.write_bytes(to_json(artifact, indent=2))


class RecordingSearch(BookSearch):
    """Capture evidence at the real search boundary, including repeated searches."""

    def __init__(self, searcher: BookSearch) -> None:
        self.searcher = searcher
        self.book_id = searcher.book_id
        self.index_id = searcher.index_id
        self.searches: list[dict] = []

    async def stream_search(
        self, query: str
    ) -> AsyncIterator[SearchPhase | BookSearchResult]:
        async with aclosing(self.searcher.stream_search(query)) as updates:
            async for update in updates:
                if isinstance(update, BookSearchResult):
                    self.searches.append(
                        {"query": query, **update.model_dump(mode="json")}
                    )
                yield update

    def retrieval_context(self) -> list[str]:
        # Across tool calls, one chunk may appear repeatedly. Keep its first
        # occurrence; retain per-query rankings separately in the raw artifact.
        passages = {}
        for search in self.searches:
            for passage in search["passages"]:
                passages.setdefault(passage["chunk_id"], passage["text"])
        return list(passages.values())


async def run_question(
    searcher: BookSearch, model: Model, question: str, *, timeout_seconds: float
) -> dict:
    recording = RecordingSearch(searcher)
    deps = BookChatDeps(searcher=recording)
    agent = create_book_agent(model)
    started = perf_counter()
    async with (
        asyncio.timeout(timeout_seconds),
        agent.run_stream_events(question, deps=deps) as events,
    ):
        async for event in events:
            if isinstance(event, AgentRunResultEvent):
                result = event.result
                return {
                    "actual_output": result.output,
                    "retrieval_context": recording.retrieval_context(),
                    "searches": recording.searches,
                    "citations": deps.citations.snapshot().model_dump(mode="json"),
                    "book_id": str(searcher.book_id),
                    "index_id": str(searcher.index_id),
                    "latency_seconds": perf_counter() - started,
                    "usage": asdict(result.usage),
                }
    raise RuntimeError("Agent stream ended without a result")
