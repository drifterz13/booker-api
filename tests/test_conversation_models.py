import json
import unittest
from uuid import uuid4

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.ui.vercel_ai.request_types import UIMessage

from app.models.conversation import ConversationMessage, ConversationRole
from app.schemas.citation import CitationData


class ConversationMessageTests(unittest.TestCase):
    def test_citations_survive_ui_storage_separately_from_agent_history(self):
        book_id, index_id, chunk_id = uuid4(), uuid4(), uuid4()
        user = UIMessage.model_validate(
            {
                "id": "user-1",
                "role": "user",
                "parts": [{"type": "text", "text": "Why?"}],
            }
        )
        assistant = UIMessage.model_validate(
            {
                "id": "assistant-1",
                "role": "assistant",
                "parts": [
                    {
                        "type": "data-citations",
                        "id": "citations",
                        "data": {
                            "sources": [
                                {
                                    "id": "s1",
                                    "book_id": str(book_id),
                                    "index_id": str(index_id),
                                    "chunk_id": str(chunk_id),
                                    "pdf_pages": [3],
                                    "section_path": ["Habits"],
                                }
                            ]
                        },
                    },
                    {"type": "text", "text": "Practice helps [1](#cite-s1)."},
                ],
            }
        )
        agent_history = VercelAIAdapter.load_messages([user, assistant])
        self.assertNotIn(
            "data-citations",
            [
                part.type
                for message in VercelAIAdapter.dump_messages(agent_history)
                for part in message.parts
            ],
        )

        message = ConversationMessage(
            conversation_id=uuid4(),
            position=1,
            role=ConversationRole.ASSISTANT,
            ui_message=assistant.model_dump(mode="json"),
            model_messages=ModelMessagesTypeAdapter.dump_python(
                agent_history, mode="json"
            ),
        )
        # JSON round-trip mirrors the PostgreSQL JSONB boundary.
        stored_ui = UIMessage.model_validate(json.loads(json.dumps(message.ui_message)))
        stored_history = ModelMessagesTypeAdapter.validate_python(
            json.loads(json.dumps(message.model_messages))
        )

        citation_part = next(
            part for part in stored_ui.parts if part.type == "data-citations"
        )
        source = CitationData.model_validate(citation_part.data).sources[0]
        self.assertEqual(source.book_id, book_id)
        self.assertEqual(source.index_id, index_id)
        self.assertEqual(source.chunk_id, chunk_id)
        self.assertEqual(source.pdf_pages, [3])
        self.assertEqual(len(stored_history), len(agent_history))
