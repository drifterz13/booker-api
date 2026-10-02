from uuid import UUID

from sqlmodel import Session, select

from ..models.book import Book
from ..models.conversation import Conversation, ConversationMessage


class ConversationService:
    """Read public conversations and their client-visible messages."""

    def __init__(self, *, session: Session) -> None:
        self._session = session

    def book_exists(self, book_id: UUID) -> bool:
        return self._session.get(Book, book_id) is not None

    def list_for_book(
        self, book_id: UUID, *, offset: int, limit: int
    ) -> list[Conversation]:
        return list(
            self._session.exec(
                select(Conversation)
                .where(Conversation.book_id == book_id)
                .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
                .offset(offset)
                .limit(limit)
            ).all()
        )

    def get(self, book_id: UUID, conversation_id: UUID) -> Conversation | None:
        return self._session.exec(
            select(Conversation).where(
                Conversation.id == conversation_id, Conversation.book_id == book_id
            )
        ).first()

    def messages(self, conversation_id: UUID) -> list[ConversationMessage]:
        return list(
            self._session.exec(
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == conversation_id)
                .order_by(ConversationMessage.position)
            ).all()
        )
