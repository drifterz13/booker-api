import json
import unittest
from unittest.mock import patch

from pydantic_ai.ui.vercel_ai.request_types import UIMessage

from app.services.conversations import recent_messages


class ConversationContextTests(unittest.TestCase):
    def test_only_complete_old_turns_leave_model_context(self):
        messages = [
            UIMessage.model_validate(
                {
                    "id": str(index),
                    "role": "user" if index % 2 == 0 else "assistant",
                    "parts": [{"type": "text", "text": text}],
                }
            )
            for index, text in enumerate(
                ["Old question", "Old answer", "New question", "New answer"]
            )
        ]
        budget = sum(
            len(json.dumps(message.model_dump(mode="json")))
            for message in messages[-2:]
        )
        with patch("app.services.conversations.MAX_CONTEXT_CHARS", budget):
            context = recent_messages(messages)

        self.assertEqual([message.id for message in context], ["2", "3"])
        self.assertEqual(len(messages), 4)
