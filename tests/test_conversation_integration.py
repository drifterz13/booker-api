import unittest
from uuid import uuid4

from pydantic_ai.ui.vercel_ai.request_types import DataUIPart, UIMessage
from sqlalchemy import delete
from sqlmodel import Session, select

from app.models.book import Book
from app.models.conversation import Conversation, ConversationMessage, ConversationRole
from tests.support import database_resources


class ConversationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.engine = cls.enterClassContext(database_resources())

    def test_cited_message_survives_database_round_trip(self):
        book = Book(filename="conversation-test.pdf", object_key=f"books/{uuid4()}.pdf")
        conversation = Conversation(book_id=book.id, title="Habits")
        book_id = book.id
        conversation_id = conversation.id
        citation = {
            "id": "s1",
            "book_id": str(book.id),
            "index_id": str(uuid4()),
            "chunk_id": str(uuid4()),
            "pdf_pages": [3],
            "section_path": ["Habits"],
        }
        ui_message = {
            "id": "assistant-1",
            "role": "assistant",
            "parts": [
                {
                    "type": "data-citations",
                    "id": "citations",
                    "data": {"sources": [citation]},
                },
                {"type": "text", "text": "Practice helps [1](#cite-s1)."},
            ],
        }
        message = ConversationMessage(
            conversation_id=conversation.id,
            position=1,
            role=ConversationRole.ASSISTANT,
            ui_message=ui_message,
        )
        with Session(self.engine) as session:
            session.add_all([book, conversation, message])
            session.commit()
        self.addCleanup(self.delete_book, book_id)

        with Session(self.engine) as session:
            saved = session.exec(
                select(ConversationMessage).where(
                    ConversationMessage.conversation_id == conversation_id
                )
            ).one()
            restored = UIMessage.model_validate(saved.ui_message)
            self.assertEqual(saved.role, ConversationRole.ASSISTANT)
            citation_part = next(
                part for part in restored.parts if isinstance(part, DataUIPart)
            )
            self.assertEqual(citation_part.data["sources"], [citation])

    def delete_book(self, book_id):
        with Session(self.engine) as session:
            session.exec(delete(Book).where(Book.id == book_id))
            session.commit()
