from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

from ..dependencies import BookSearchDep, ChatConfigDep, ChatModelDep, ObservabilityDep
from ..schemas.chat import ChatRequest
from ..services.ai.chat import create_book_agent
from ..services.ai.ui_stream import stream_chat

router = APIRouter(prefix="/books", tags=["chat"])


@router.post(
    "/{book_id}/chat",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def chat_book(
    body: ChatRequest,
    searcher: BookSearchDep,
    model: ChatModelDep,
    config: ChatConfigDep,
    observability: ObservabilityDep,
) -> StreamingResponse:
    adapter = VercelAIAdapter(
        agent=create_book_agent(model, instrument=observability.instrumentation),
        run_input=body,
    )
    try:
        _ = adapter.messages  # Validate history conversion before sending SSE headers.
    except ValueError as error:
        raise HTTPException(status_code=422, detail="Invalid chat history") from error
    return StreamingResponse(
        stream_chat(
            adapter,
            searcher,
            timeout_seconds=config.chat_timeout_seconds,
            observability=observability,
            session_id=body.id,
        ),
        media_type="text/event-stream",
        headers={
            **adapter.build_event_stream().response_headers,
            "cache-control": "no-cache",
            "x-accel-buffering": "no",
        },
    )
