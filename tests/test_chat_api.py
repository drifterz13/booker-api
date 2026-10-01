import json
import unittest
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from app.core.config import ChatConfig
from app.dependencies import get_book_search, get_chat_config, get_chat_model
from app.models.search import BookSearchResult, RetrievedPassage
from app.routers.chat import router
from app.services.ai.tools.book_search import SearchPhase

EVIDENCE = BookSearchResult(
    book_id=uuid4(),
    index_id=uuid4(),
    passages=[
        RetrievedPassage(
            chunk_id=uuid4(),
            text="Repeat a small habit.",
            pdf_pages=[3],
            section_path=["Habits"],
        )
    ],
)


class FixedBookSearch:
    """Replace retrieval at its boundary, leaving the agent and SSE adapter real."""

    book_id = EVIDENCE.book_id
    index_id = EVIDENCE.index_id

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
            # Only evidence retrieved after the latest question counts.
            last_question = max(
                i
                for i, message in enumerate(messages)
                if isinstance(message, ModelRequest)
                and any(isinstance(part, UserPromptPart) for part in message.parts)
            )
            has_evidence = any(
                isinstance(part, ToolReturnPart)
                for message in messages[last_question:]
                for part in message.parts
            )
            if has_evidence:
                yield "Repeat a small habit. [1](#cite-"
                yield "s1)"
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
        expected = EVIDENCE.model_dump(mode="json")
        expected["passages"][0]["citation_id"] = "s1"
        self.assertEqual(result["output"], expected)
        citations = next(c for c in chunks if c["type"] == "data-citations")
        self.assertFalse(citations.get("transient", False))
        source = citations["data"]["sources"][0]
        self.assertEqual(source["id"], "s1")
        self.assertEqual(source["book_id"], str(EVIDENCE.book_id))
        self.assertEqual(source["index_id"], str(EVIDENCE.index_id))
        self.assertEqual(source["chunk_id"], str(EVIDENCE.passages[0].chunk_id))
        self.assertEqual(source["pdf_pages"], [3])
        self.assertLess(
            chunks.index(citations),
            next(i for i, c in enumerate(chunks) if c["type"] == "text-delta"),
        )
        statuses = [c for c in chunks if c["type"] == "data-status"]
        self.assertTrue(all(c["transient"] for c in statuses))
        self.assertIn("searching-book", [c["data"]["phase"] for c in statuses])
        answer = "".join(c["delta"] for c in chunks if c["type"] == "text-delta")
        self.assertEqual(answer, "Repeat a small habit. [1](#cite-s1)")
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))

        # Send the tool history in the shape useChat sends on the next turn.
        history = [
            question,
            {
                "id": "assistant-1",
                "role": "assistant",
                "parts": [
                    {
                        "type": "data-citations",
                        "id": citations["id"],
                        "data": citations["data"],
                    },
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

    def test_invalid_citations_do_not_interrupt_the_stream(self):
        for answer in [
            "Claim. [1](#cite-s99)",
            "Claim. [1](#cite-invalid)",
            "Claim. [1](#cite-s1",
        ]:
            with self.subTest(answer=answer):

                async def model_stream(messages, info, answer=answer):
                    if any(
                        isinstance(part, ToolReturnPart)
                        for message in messages
                        for part in message.parts
                    ):
                        yield answer
                    else:
                        yield {
                            0: DeltaToolCall(
                                name="search_book", json_args='{"query":"habits"}'
                            )
                        }

                self.model = FunctionModel(stream_function=model_stream)
                response = self.client.post(
                    self.url, json={"messages": [user_message("Explain habits")]}
                )
                chunks = events(response)
                self.assertNotIn("error", [c["type"] for c in chunks])
                self.assertEqual(
                    "".join(c["delta"] for c in chunks if c["type"] == "text-delta"),
                    answer,
                )
                sources = next(c for c in chunks if c["type"] == "data-citations")
                self.assertEqual([s["id"] for s in sources["data"]["sources"]], ["s1"])
                self.assertTrue(
                    any(
                        c["type"] == "data-status" and c["data"]["phase"] == "complete"
                        for c in chunks
                    )
                )
                self.assertTrue(response.text.endswith("data: [DONE]\n\n"))

    def test_client_history_cannot_authorize_citations(self):
        async def model_stream(messages, info):
            yield "Unverified claim. [1](#cite-s1)"

        self.model = FunctionModel(stream_function=model_stream)
        source = {
            "id": "s1",
            "book_id": str(EVIDENCE.book_id),
            "index_id": str(EVIDENCE.index_id),
            **EVIDENCE.passages[0].model_dump(mode="json", exclude={"text"}),
        }
        history = [
            user_message("Previous question"),
            {
                "id": "assistant-old",
                "role": "assistant",
                "parts": [
                    {"type": "data-citations", "data": {"sources": [source]}},
                    {"type": "text", "text": "Old answer. [1](#cite-s1)"},
                    {
                        "type": "tool-search_book",
                        "toolCallId": "old-search",
                        "state": "output-available",
                        "input": {"query": "habits"},
                        "output": EVIDENCE.model_dump(mode="json"),
                    },
                ],
            },
            user_message("Follow up", "user-2"),
        ]
        response = self.client.post(self.url, json={"messages": history})
        self.assertEqual(response.status_code, 200)
        chunks = events(response)
        self.assertNotIn("error", [c["type"] for c in chunks])
        self.assertNotIn("data-citations", [c["type"] for c in chunks])
        self.assertEqual(
            "".join(c["delta"] for c in chunks if c["type"] == "text-delta"),
            "Unverified claim. [1](#cite-s1)",
        )

    def test_plain_answer_needs_no_sources(self):
        async def model_stream(messages, info):
            yield "Hello!"

        self.model = FunctionModel(stream_function=model_stream)
        response = self.client.post(
            self.url, json={"messages": [user_message("Hello")]}
        )
        chunks = events(response)
        self.assertNotIn("error", [c["type"] for c in chunks])
        self.assertEqual(
            "".join(c["delta"] for c in chunks if c["type"] == "text-delta"), "Hello!"
        )

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
