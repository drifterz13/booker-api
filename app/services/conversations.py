import json
from datetime import UTC, datetime
from uuid import UUID

from pydantic_ai.ui.vercel_ai.request_types import TextUIPart, UIMessage
from sqlalchemy import func
from sqlalchemy.engine import Engine
from sqlmodel import Session, col, select

from ..models.book import Book
from ..models.conversation import Conversation, ConversationMessage, ConversationRole

MAX_CONTEXT_CHARS = 80_000


def recent_messages(messages: list[UIMessage]) -> list[UIMessage]:
    """Drop oldest complete user/assistant turns from model context only."""
    size = sum(len(json.dumps(message.model_dump(mode="json"))) for message in messages)
    while len(messages) > 2 and size > MAX_CONTEXT_CHARS:
        removed = messages[:2]
        messages = messages[2:]
        size -= sum(
            len(json.dumps(message.model_dump(mode="json"))) for message in removed
        )
    return messages


class ConversationService:
    """Read and save public conversations with short database sessions."""

    def __init__(self, *, engine: Engine) -> None:
        self._engine = engine

    def book_exists(self, book_id: UUID) -> bool:
        with Session(self._engine) as session:
            return session.get(Book, book_id) is not None

    def list_conversations(
        self, *, offset: int, limit: int, book_id: UUID | None = None
    ) -> list[Conversation]:
        with Session(self._engine) as session:
            query = select(Conversation)
            if book_id is not None:
                query = query.where(Conversation.book_id == book_id)
            return list(
                session.exec(
                    query.order_by(
                        col(Conversation.updated_at).desc(), col(Conversation.id).desc()
                    )
                    .offset(offset)
                    .limit(limit)
                ).all()
            )

    def get(self, book_id: UUID, conversation_id: UUID) -> Conversation | None:
        with Session(self._engine) as session:
            return session.exec(
                select(Conversation).where(
                    Conversation.id == conversation_id, Conversation.book_id == book_id
                )
            ).first()

    def messages(self, conversation_id: UUID) -> list[ConversationMessage]:
        with Session(self._engine) as session:
            return list(
                session.exec(
                    select(ConversationMessage)
                    .where(ConversationMessage.conversation_id == conversation_id)
                    .order_by(col(ConversationMessage.position))
                ).all()
            )

    def load_context(
        self, book_id: UUID, conversation_id: UUID
    ) -> list[UIMessage] | None:
        if self.get(book_id, conversation_id) is None:
            return None
        messages = [
            UIMessage.model_validate(row.ui_message)
            for row in self.messages(conversation_id)
        ]
        return recent_messages(messages)

    def save_turn(
        self,
        *,
        book_id: UUID,
        conversation_id: UUID,
        user_message: UIMessage,
        assistant_message: UIMessage,
    ) -> None:
        with Session(self._engine) as session, session.begin():
            conversation = session.get(
                Conversation, conversation_id, with_for_update=True
            )
            if conversation is None:
                title = " ".join(
                    part.text
                    for part in user_message.parts
                    if isinstance(part, TextUIPart)
                ).strip()[:80]
                conversation = Conversation(
                    id=conversation_id, book_id=book_id, title=title
                )
                session.add(conversation)
                session.flush()
            else:
                if conversation.book_id != book_id:
                    raise ValueError("Conversation belongs to another book")
                # Inserting a child message does not update its parent row.
                conversation.updated_at = datetime.now(UTC)
                session.add(conversation)

            last_position = session.exec(
                select(func.max(ConversationMessage.position)).where(
                    ConversationMessage.conversation_id == conversation_id
                )
            ).one()
            position = (last_position if last_position is not None else -1) + 1
            session.add_all(
                [
                    ConversationMessage(
                        conversation_id=conversation_id,
                        position=position,
                        role=ConversationRole.USER,
                        ui_message=user_message.model_dump(
                            mode="json", by_alias=True, exclude_none=True
                        ),
                    ),
                    ConversationMessage(
                        conversation_id=conversation_id,
                        position=position + 1,
                        role=ConversationRole.ASSISTANT,
                        ui_message=assistant_message.model_dump(
                            mode="json", by_alias=True, exclude_none=True
                        ),
                    ),
                ]
            )
