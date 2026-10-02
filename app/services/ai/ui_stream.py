import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing

from opentelemetry.trace import StatusCode
from pydantic_ai import AgentRunResult
from pydantic_ai.messages import ModelMessage
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.ui.vercel_ai.request_types import DataUIPart, UIMessage
from pydantic_ai.ui.vercel_ai.response_types import (
    DataChunk,
    DoneChunk,
    ErrorChunk,
    FinishChunk,
    StartStepChunk,
)

from ..observability import NO_OBSERVABILITY, Observability
from .chat import STATUS_MESSAGES, BookChatDeps
from .tools.book_search import BookSearch

logger = logging.getLogger(__name__)


def status_chunk(phase: str) -> DataChunk:
    return DataChunk(
        type="data-status",
        data={"phase": phase, "message": STATUS_MESSAGES[phase]},
        transient=True,
    )


def assistant_ui_message(
    result: AgentRunResult[str], citations: list[DataChunk], message_id: str
) -> UIMessage:
    """Build the assistant message to restore in the chat UI."""
    assistant_messages = [
        message
        for message in VercelAIAdapter.dump_messages(result.new_messages())
        if message.role == "assistant"
    ]
    citation_parts = (
        [
            DataUIPart(
                type=citations[-1].type,
                id=citations[-1].id,
                data=citations[-1].data,
            )
        ]
        if citations
        else []
    )
    return assistant_messages[-1].model_copy(
        update={
            "id": message_id,
            "parts": [
                *citation_parts,
                *(part for message in assistant_messages for part in message.parts),
            ],
        }
    )


async def stream_chat(
    adapter: VercelAIAdapter[BookChatDeps, str],
    searcher: BookSearch,
    *,
    timeout_seconds: float,
    observability: Observability = NO_OBSERVABILITY,
    session_id: str = "",
    message_history: list[ModelMessage] | None = None,
    on_complete: Callable[[AgentRunResult[str], list[DataChunk]], Awaitable[None]]
    | None = None,
) -> AsyncIterator[str]:
    event_stream = adapter.build_event_stream()
    deps = BookChatDeps(searcher=searcher)
    citation_chunks: list[DataChunk] = []

    async def completed(result: AgentRunResult[str]) -> None:
        if on_complete is not None:
            await on_complete(result, citation_chunks)

    with observability.chat(
        session_id=session_id,
        book_id=str(searcher.book_id),
        index_id=str(searcher.index_id),
    ) as span:
        try:
            async with (
                asyncio.timeout(timeout_seconds),
                aclosing(
                    adapter.run_stream_native(
                        deps=deps,
                        message_history=message_history,
                        conversation_id=session_id or None,
                    )
                ) as native,
                aclosing(
                    event_stream.transform_stream(
                        native, on_complete=completed if on_complete else None
                    )
                ) as chunks,
            ):
                async for chunk in chunks:
                    if isinstance(chunk, ErrorChunk):
                        span.set_status(StatusCode.ERROR, "Chat failed")
                        # The adapter's default error contains exception text.
                        logger.error("Book chat failed: %s", chunk.error_text)
                        chunk = ErrorChunk(error_text="Could not generate an answer.")
                    elif isinstance(chunk, DataChunk) and chunk.type == "data-status":
                        chunk = chunk.model_copy(update={"transient": True})
                    elif (
                        isinstance(chunk, DataChunk) and chunk.type == "data-citations"
                    ):
                        citation_chunks.append(chunk)
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
        except (asyncio.CancelledError, GeneratorExit):
            span.set_status(StatusCode.ERROR, "Chat cancelled")
            raise
        except TimeoutError:
            span.set_status(StatusCode.ERROR, "Chat timed out")
            yield event_stream.encode_event(
                ErrorChunk(error_text="The chat request timed out.")
            )
            yield event_stream.encode_event(DoneChunk())
        except Exception:
            span.set_status(StatusCode.ERROR, "Chat failed")
            logger.exception("Book chat failed")
            yield event_stream.encode_event(
                ErrorChunk(error_text="Could not generate an answer.")
            )
            yield event_stream.encode_event(DoneChunk())
