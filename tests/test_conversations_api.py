import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlmodel import Session

from app.db.database import get_session
from app.models.book import Book
from app.models.conversation import Conversation, ConversationMessage, ConversationRole
from app.routers.conversations import router
from tests.support import database_resources


class ConversationsApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.engine = cls.enterClassContext(database_resources())

    def setUp(self):
        now = datetime.now(UTC)
        book = Book(filename="conversation-api.pdf", object_key=f"books/{uuid4()}.pdf")
        other_book = Book(filename="other-api.pdf", object_key=f"books/{uuid4()}.pdf")
        older = Conversation(book_id=book.id, title="Older", updated_at=now)
        newer = Conversation(
            book_id=book.id, title="Newer", updated_at=now + timedelta(minutes=1)
        )
        another_book = Conversation(book_id=other_book.id, title="Other book")
        self.book_id = book.id
        self.other_book_id = other_book.id
        self.older_id = older.id
        self.newer_id = newer.id
        self.another_book_id = another_book.id
        self.addCleanup(self.delete_books)

        citation = {
            "id": "s1",
            "book_id": str(book.id),
            "index_id": str(uuid4()),
            "chunk_id": str(uuid4()),
            "pdf_pages": [3],
            "section_path": ["Habits"],
        }
        self.citation = citation
        user = ConversationMessage(
            conversation_id=newer.id,
            position=0,
            role=ConversationRole.USER,
            ui_message={
                "id": "user-1",
                "role": "user",
                "parts": [{"type": "text", "text": "Why?"}],
            },
        )
        assistant = ConversationMessage(
            conversation_id=newer.id,
            position=1,
            role=ConversationRole.ASSISTANT,
            ui_message={
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
            },
            model_messages=[{"internal": "do not expose"}],
        )
        with Session(self.engine) as session:
            session.add_all([book, other_book, older, newer, another_book])
            session.flush()
            session.add_all([assistant, user])
            session.commit()

        app = FastAPI()
        app.include_router(router)

        def test_session():
            with Session(self.engine) as session:
                yield session

        app.dependency_overrides[get_session] = test_session
        self.client = self.enterContext(TestClient(app))

    def delete_books(self):
        with Session(self.engine) as session:
            session.exec(
                delete(Book).where(Book.id.in_([self.book_id, self.other_book_id]))
            )
            session.commit()

    def test_list_is_book_scoped_newest_first_and_paginated(self):
        url = f"/books/{self.book_id}/conversations"
        first = self.client.get(url, params={"limit": 1}).json()
        second = self.client.get(url, params={"limit": 1, "offset": 1}).json()
        self.assertEqual([item["id"] for item in first], [str(self.newer_id)])
        self.assertEqual([item["id"] for item in second], [str(self.older_id)])
        self.assertNotIn("messages", first[0])
        self.assertEqual(self.client.get(url, params={"limit": 0}).status_code, 422)
        self.assertEqual(
            self.client.get(f"/books/{uuid4()}/conversations").status_code, 404
        )

    def test_detail_returns_ordered_ui_messages_and_citations(self):
        response = self.client.get(
            f"/books/{self.book_id}/conversations/{self.newer_id}"
        )
        self.assertEqual(response.status_code, 200, response.text)
        detail = response.json()
        self.assertEqual(
            [item["id"] for item in detail["messages"]], ["user-1", "assistant-1"]
        )
        self.assertEqual(
            detail["messages"][1]["parts"][0]["data"]["sources"],
            [self.citation],
        )
        self.assertNotIn("model_messages", detail)
        self.assertEqual(
            self.client.get(
                f"/books/{self.other_book_id}/conversations/{self.newer_id}"
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(
                f"/books/{self.book_id}/conversations/{uuid4()}"
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(
                f"/books/{self.book_id}/conversations/{self.older_id}"
            ).json()["messages"],
            [],
        )
