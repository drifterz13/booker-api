from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic_ai import AgentRunResult
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.ui.vercel_ai.request_types import UIMessage
from pydantic_ai.ui.vercel_ai.response_types import DataChunk
from sqlalchemy.exc import SQLAlchemyError

from ..dependencies import (
    BookSearchDep,
    ChatConfigDep,
    ChatModelDep,
    ConversationServiceDep,
    ObservabilityDep,
)
from ..schemas.chat import ConversationChatRequest
from ..schemas.conversation import (
    ConversationDetail,
    ConversationListParams,
    ConversationPublic,
)
from ..services.ai.chat import create_book_agent
from ..services.ai.ui_stream import assistant_ui_message, stream_chat
from ..services.conversations import ConversationService

router = APIRouter(prefix="/books/{book_id}/conversations", tags=["conversations"])


def conversation_chat_response(
    *,
    body: ConversationChatRequest,
    book_id: UUID,
    conversation_id: UUID,
    service: ConversationService,
    searcher: BookSearchDep,
    model: ChatModelDep,
    config: ChatConfigDep,
    observability: ObservabilityDep,
    existing: bool,
) -> StreamingResponse:
    context = service.load_context(book_id, conversation_id) if existing else []
    if context is None:
        raise HTTPException(404, "Conversation does not exist")
    # ChatRequest requires the newest message to be the user's question.
    user_message = body.messages[-1]
    assistant_id = str(uuid4())
    run_input = body.model_copy(update={"id": str(conversation_id)})
    adapter = VercelAIAdapter(
        agent=create_book_agent(model, instrument=observability.instrumentation),
        run_input=run_input,
        server_message_id=assistant_id,
    )

    async def save_answer(
        result: AgentRunResult[str], citations: list[DataChunk]
    ) -> None:
        assistant_message = assistant_ui_message(result, citations, assistant_id)
        await run_in_threadpool(
            service.save_turn,
            book_id=book_id,
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )

    return StreamingResponse(
        stream_chat(
            adapter,
            searcher,
            timeout_seconds=config.chat_timeout_seconds,
            observability=observability,
            session_id=str(conversation_id),
            message_history=VercelAIAdapter.load_messages(context),
            on_complete=save_answer,
        ),
        media_type="text/event-stream",
        headers={
            **(adapter.build_event_stream().response_headers or {}),
            "cache-control": "no-cache",
            "x-accel-buffering": "no",
            "x-conversation-id": str(conversation_id),
        },
    )


@router.post(
    "",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
def start_conversation(
    body: ConversationChatRequest,
    book_id: UUID,
    searcher: BookSearchDep,
    model: ChatModelDep,
    config: ChatConfigDep,
    observability: ObservabilityDep,
    service: ConversationServiceDep,
) -> StreamingResponse:
    return conversation_chat_response(
        body=body,
        book_id=book_id,
        conversation_id=uuid4(),
        service=service,
        searcher=searcher,
        model=model,
        config=config,
        observability=observability,
        existing=False,
    )


@router.post(
    "/{conversation_id}/messages",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
def continue_conversation(
    body: ConversationChatRequest,
    book_id: UUID,
    conversation_id: UUID,
    searcher: BookSearchDep,
    model: ChatModelDep,
    config: ChatConfigDep,
    observability: ObservabilityDep,
    service: ConversationServiceDep,
) -> StreamingResponse:
    return conversation_chat_response(
        body=body,
        book_id=book_id,
        conversation_id=conversation_id,
        service=service,
        searcher=searcher,
        model=model,
        config=config,
        observability=observability,
        existing=True,
    )


@router.get("")
def list_conversations(
    book_id: UUID,
    params: Annotated[ConversationListParams, Query()],
    service: ConversationServiceDep,
) -> list[ConversationPublic]:
    try:
        if not service.book_exists(book_id):
            raise HTTPException(404, "Book does not exist")
        conversations = service.list_for_book(
            book_id, offset=params.offset, limit=params.limit
        )
    except SQLAlchemyError as error:
        raise HTTPException(503, "Could not list conversations") from error
    return [ConversationPublic.model_validate(item) for item in conversations]


@router.get("/{conversation_id}")
def get_conversation(
    book_id: UUID, conversation_id: UUID, service: ConversationServiceDep
) -> ConversationDetail:
    try:
        conversation = service.get(book_id, conversation_id)
        if conversation is None:
            raise HTTPException(404, "Conversation does not exist")
        messages = service.messages(conversation_id)
    except SQLAlchemyError as error:
        raise HTTPException(503, "Could not load the conversation") from error
    return ConversationDetail(
        **ConversationPublic.model_validate(conversation).model_dump(),
        messages=[UIMessage.model_validate(message.ui_message) for message in messages],
    )
