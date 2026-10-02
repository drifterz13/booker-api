from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic_ai.ui.vercel_ai.request_types import UIMessage
from sqlalchemy.exc import SQLAlchemyError

from ..dependencies import ConversationServiceDep
from ..schemas.conversation import (
    ConversationDetail,
    ConversationListParams,
    ConversationPublic,
)

router = APIRouter(prefix="/books/{book_id}/conversations", tags=["conversations"])


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
