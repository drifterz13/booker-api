import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing

from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.ui.vercel_ai.response_types import (
    DataChunk,
    DoneChunk,
    ErrorChunk,
    FinishChunk,
    StartStepChunk,
)

from .chat import STATUS_MESSAGES, BookChatDeps
from .tools.book_search import BookSearch

logger = logging.getLogger(__name__)


def status_chunk(phase: str) -> DataChunk:
    return DataChunk(
        type="data-status",
        data={"phase": phase, "message": STATUS_MESSAGES[phase]},
        transient=True,
    )


async def stream_chat(
    adapter: VercelAIAdapter[BookChatDeps, str],
    searcher: BookSearch,
    *,
    timeout_seconds: float,
) -> AsyncIterator[str]:
    event_stream = adapter.build_event_stream()
    deps = BookChatDeps(searcher=searcher)
    try:
        async with asyncio.timeout(timeout_seconds):
            async with (
                aclosing(adapter.run_stream_native(deps=deps)) as native,
                aclosing(event_stream.transform_stream(native)) as chunks,
            ):
                async for chunk in chunks:
                    if isinstance(chunk, ErrorChunk):
                        # The adapter's default error contains exception text.
                        logger.error("Book chat failed: %s", chunk.error_text)
                        chunk = ErrorChunk(error_text="Could not generate an answer.")
                    elif isinstance(chunk, DataChunk) and chunk.type == "data-status":
                        chunk = chunk.model_copy(update={"transient": True})
                    elif (
                        isinstance(chunk, FinishChunk)
                        and chunk.finish_reason != "error"
                    ):
                        yield event_stream.encode_event(status_chunk("complete"))
                    yield event_stream.encode_event(chunk)
                    if isinstance(chunk, StartStepChunk):
                        yield event_stream.encode_event(
                            status_chunk("generating-answer")
                        )
    except TimeoutError:
        yield event_stream.encode_event(
            ErrorChunk(error_text="The chat request timed out.")
        )
        yield event_stream.encode_event(DoneChunk())
    except Exception:
        logger.exception("Book chat failed")
        yield event_stream.encode_event(
            ErrorChunk(error_text="Could not generate an answer.")
        )
        yield event_stream.encode_event(DoneChunk())
