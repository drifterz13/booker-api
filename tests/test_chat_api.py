import json
import unittest
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from app.core.config import ChatConfig
from app.dependencies import get_book_search, get_chat_config, get_chat_model
from app.routers.chat import router
from app.services.ai.tools.book_search import SearchPhase

EVIDENCE = {"passages": [{"text": "Repeat a small habit.", "pdf_pages": [3]}]}


class FixedBookSearch:
    """Replace retrieval at its boundary, leaving the agent and SSE adapter real."""

    async def stream_search(self, query):
        yield SearchPhase.EMBEDDING_QUERY
        yield SearchPhase.SEARCHING_BOOK
        yield EVIDENCE


def user_message(text, message_id="user-1"):
    return {"id": message_id, "role": "user", "parts": [{"type": "text", "text": text}]}


def events(response):
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        async def model_stream(messages, info):
            has_evidence = any(
                isinstance(part, ToolReturnPart)
                for message in messages
                for part in message.parts
            )
            if has_evidence:
                yield "Repeat a small habit. [PDF pages: 3]"
            else:
                yield {
                    0: DeltaToolCall(name="search_book", json_args='{"query":"habits"}')
                }

        self.model = FunctionModel(stream_function=model_stream)
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_book_search] = FixedBookSearch
        app.dependency_overrides[get_chat_model] = lambda: self.model
        app.dependency_overrides[get_chat_config] = lambda: ChatConfig(
            _env_file=None, openai_api_key="test-key"
        )
        self.client = self.enterContext(TestClient(app))
        self.url = f"/books/{uuid4()}/chat"

    def test_tool_progress_answer_and_follow_up(self):
        question = user_message("How do habits form?")
        response = self.client.post(self.url, json={"messages": [question]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-vercel-ai-ui-message-stream"], "v1")
        chunks = events(response)
        tool_input = next(c for c in chunks if c["type"] == "tool-input-available")
        result = next(c for c in chunks if c["type"] == "tool-output-available")
        self.assertEqual(result["output"], EVIDENCE)
        statuses = [c for c in chunks if c["type"] == "data-status"]
        self.assertTrue(all(c["transient"] for c in statuses))
        self.assertIn("searching-book", [c["data"]["phase"] for c in statuses])
        answer = "".join(c["delta"] for c in chunks if c["type"] == "text-delta")
        self.assertEqual(answer, "Repeat a small habit. [PDF pages: 3]")
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))

        # Send the tool history in the shape useChat sends on the next turn.
        history = [
            question,
            {
                "id": "assistant-1",
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool-search_book",
                        "toolCallId": tool_input["toolCallId"],
                        "state": "output-available",
                        "input": tool_input["input"],
                        "output": result["output"],
                    },
                    {"type": "text", "text": answer},
                ],
            },
            user_message("Explain more", "user-2"),
        ]
        follow_up = self.client.post(self.url, json={"messages": history})
        self.assertEqual(follow_up.status_code, 200)
        self.assertNotIn("error", [c["type"] for c in events(follow_up)])

    def test_provider_errors_are_not_exposed_to_the_client(self):
        async def failing_stream(messages, info):
            raise RuntimeError("private provider credentials")
            yield "unreachable"

        self.model = FunctionModel(stream_function=failing_stream)
        with self.assertLogs("app.services.ai.ui_stream", level="ERROR"):
            response = self.client.post(
                self.url, json={"messages": [user_message("Hello")]}
            )
        self.assertIn(
            {"type": "error", "errorText": "Could not generate an answer."},
            events(response),
        )
        self.assertNotIn("private provider credentials", response.text)
