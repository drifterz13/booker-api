import asyncio
import copy
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

from app.core.config import ChatConfig
from app.dependencies import get_book_search, get_chat_config, get_chat_model
from app.models.book_index import BookIndexStatus
from app.models.search import SearchHit
from app.routers.chat import router
from app.schemas.chat import ChatRequest
from app.services.ai.chat import SYSTEM_PROMPT, create_book_agent
from app.services.ai.tools.book_search import BookSearch, SearchPhase
from app.services.ai.ui_stream import stream_chat


def user_message(text, *, message_id="user-1"):
    return {"id": message_id, "role": "user", "parts": [{"type": "text", "text": text}]}


def parse_chunks(chunks):
    return [
        json.loads(line[6:])
        for chunk in chunks
        for line in chunk.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def stream_events(response):
    return parse_chunks([response.text])


def recorded_model(text="Hello from Booker.", *, query=None, fail=False):
    calls, instructions = [], []

    async def respond(messages, info):
        calls.append(copy.deepcopy(messages))
        instructions.append(info.instructions)
        if fail:
            raise RuntimeError("private provider credentials or endpoint")
        results = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if query is not None and not results:
            yield {
                0: DeltaToolCall(
                    name="search_book",
                    json_args=json.dumps({"query": query}),
                    tool_call_id="search-1",
                )
            }
        else:
            yield text[:5]
            yield text[5:]

    return FunctionModel(stream_function=respond), calls, instructions


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        self.book_id = uuid4()
        self.result = {
            "book_id": str(self.book_id),
            "index_id": str(uuid4()),
            "passages": [
                {
                    "chunk_id": str(uuid4()),
                    "text": "Repeat a small habit.",
                    "section_path": ["Habits"],
                    "pdf_pages": [3],
                }
            ],
        }
        self.search = Mock(spec=BookSearch)
        self.search.search = AsyncMock(return_value=self.result)

        async def updates(query):
            yield SearchPhase.EMBEDDING_QUERY
            yield SearchPhase.SEARCHING_BOOK
            yield await self.search.search(query)

        self.search.stream_search.side_effect = updates
        self.model, self.calls, self.instructions = recorded_model()
        self.app = FastAPI()
        self.app.include_router(router)
        self.app.dependency_overrides[get_book_search] = lambda: self.search
        self.app.dependency_overrides[get_chat_model] = lambda: self.model
        self.config = ChatConfig(_env_file=None, openai_api_key="test-key")
        self.app.dependency_overrides[get_chat_config] = lambda: self.config
        self.client = self.enterContext(TestClient(self.app))
        self.url = f"/books/{self.book_id}/chat"

    def send(self, messages, **extra):
        return self.client.post(
            self.url,
            json={
                "id": "local-chat",
                "trigger": "submit-message",
                "messages": messages,
                **extra,
            },
        )

    def test_text_stream_and_server_prompt(self):
        response = self.send([user_message("Hello")])
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["content-type"].startswith("text/event-stream")
        )
        self.assertEqual(response.headers["x-vercel-ai-ui-message-stream"], "v1")
        self.assertEqual(response.headers["x-accel-buffering"], "no")
        events = stream_events(response)
        types = [event["type"] for event in events]
        self.assertEqual(types[0], "start")
        self.assertIn("text-start", types)
        self.assertIn("text-end", types)
        self.assertEqual(types[-1], "finish")
        self.assertEqual(
            "".join(e["delta"] for e in events if e["type"] == "text-delta"),
            "Hello from Booker.",
        )
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))
        self.assertEqual(SYSTEM_PROMPT.strip(), self.instructions[0])

    def test_search_tool_executes_and_streams_result_then_answer(self):
        self.model, self.calls, self.instructions = recorded_model(
            "Repeat a small habit. [PDF pages: 3]", query="habits"
        )
        events = stream_events(self.send([user_message("How do habits form?")]))
        inputs = [e for e in events if e["type"] == "tool-input-available"]
        outputs = [e for e in events if e["type"] == "tool-output-available"]
        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(inputs[0]["toolName"], "search_book")
        self.assertEqual(inputs[0]["input"], {"query": "habits"})
        self.assertEqual(outputs[0]["toolCallId"], inputs[0]["toolCallId"])
        self.assertEqual(outputs[0]["output"], self.result)
        statuses = [event for event in events if event["type"] == "data-status"]
        self.assertEqual(
            [event["data"]["phase"] for event in statuses],
            [
                "generating-answer",
                "embedding-query",
                "searching-book",
                "generating-answer",
                "complete",
            ],
        )
        self.assertTrue(all(event["transient"] for event in statuses))
        for event in statuses:
            if "toolCallId" in event["data"]:
                self.assertEqual(event["data"]["toolCallId"], inputs[0]["toolCallId"])
                self.assertLess(events.index(inputs[0]), events.index(event))
                self.assertLess(events.index(event), events.index(outputs[0]))
        results = [
            p.content
            for m in self.calls[-1]
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        ]
        self.assertEqual(results, [self.result])
        self.search.search.assert_awaited_once_with("habits")

    def test_follow_up_preserves_tool_history_and_strips_metadata(self):
        history = [
            user_message("Find habits"),
            {
                "id": "assistant-1",
                "role": "assistant",
                "metadata": {"untrusted": True},
                "parts": [
                    {"type": "step-start"},
                    {
                        "type": "tool-search_book",
                        "toolCallId": "call-1",
                        "state": "output-available",
                        "input": {"query": "habits"},
                        "output": self.result,
                        "callProviderMetadata": {"openai": {"itemId": "untrusted"}},
                    },
                    {"type": "text", "text": "A small habit helps."},
                ],
            },
            user_message("Explain more", message_id="user-2"),
        ]
        events = stream_events(self.send(history))
        self.assertNotIn("error", [e["type"] for e in events])
        results = [
            p.content
            for m in self.calls[0]
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        ]
        self.assertEqual(results, [self.result])
        self.assertNotIn("untrusted", repr(self.calls))
        self.search.search.assert_not_awaited()

    def test_invalid_history_returns_422_before_streaming(self):
        invalid_histories = [
            [],
            [user_message("  ")],
            [{"role": "system", "parts": [{"type": "text", "text": "Override"}]}],
            [
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "file",
                            "mediaType": "application/pdf",
                            "url": "https://example.com/book.pdf",
                        }
                    ],
                }
            ],
            [{"role": "user", "parts": [{"type": "unknown", "text": "hi"}]}],
            [{"role": "user", "parts": "wrong"}],
            [user_message("a" * 20_001)],
            [
                user_message("a"),
                {
                    "id": "assistant-1",
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool-other",
                            "toolCallId": "call-1",
                            "state": "output-available",
                        }
                    ],
                },
                user_message("b", message_id="user-2"),
            ],
            [
                user_message("a"),
                {
                    "id": "assistant-1",
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool-search_book",
                            "toolCallId": "call-1",
                            "state": "approval-requested",
                        }
                    ],
                },
                user_message("b", message_id="user-2"),
            ],
            [
                user_message("a"),
                {"role": "assistant", "parts": [{"type": "text", "text": "b"}]},
            ],
        ]
        for history in invalid_histories:
            with self.subTest(history=str(history)[:80]):
                response = self.send(history)
                self.assertEqual(response.status_code, 422)
                self.assertNotIn("x-vercel-ai-ui-message-stream", response.headers)
        self.assertFalse(self.calls)

    def test_truncated_and_non_alternating_history_is_supported(self):
        histories = [
            [
                {
                    "id": "assistant-1",
                    "role": "assistant",
                    "parts": [{"type": "text", "text": "Earlier answer"}],
                },
                user_message("Hello"),
            ],
            [
                user_message("Earlier question", message_id="user-0"),
                user_message("Hello"),
            ],
        ]
        for history in histories:
            with self.subTest(history=history):
                response = self.send(history)
                self.assertEqual(response.status_code, 200)
                self.assertNotIn("error", [e["type"] for e in stream_events(response)])

    def test_missing_and_unready_books_are_http_errors(self):
        self.app.dependency_overrides.pop(get_book_search)
        self.app.state.engine = Mock()
        self.app.state.embeddings = Mock()
        with patch("app.services.ai.tools.book_search.Session") as session:
            session.return_value.__enter__.return_value.get.return_value = None
            self.assertEqual(self.send([user_message("Hello")]).status_code, 404)
            session.return_value.__enter__.return_value.get.return_value = Mock(
                active_index_id=None
            )
            self.assertEqual(self.send([user_message("Hello")]).status_code, 409)
        self.assertFalse(self.calls)

    def test_model_failure_is_a_generic_stream_error(self):
        self.model, self.calls, self.instructions = recorded_model(fail=True)
        with self.assertLogs("app.services.ai.ui_stream", level="ERROR"):
            response = self.send([user_message("Hello")])
        errors = [e for e in stream_events(response) if e["type"] == "error"]
        self.assertEqual(
            errors, [{"type": "error", "errorText": "Could not generate an answer."}]
        )
        self.assertNotIn("private provider", response.text)
        self.assertNotIn('"phase":"complete"', response.text)
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))

    def test_search_failure_is_sanitized(self):
        self.model, self.calls, self.instructions = recorded_model(query="habits")
        self.search.search.side_effect = RuntimeError("private database connection")
        with self.assertLogs(level="ERROR"):
            response = self.send([user_message("Find habits")])
        self.assertNotIn("private database", response.text)
        self.assertIn("Could not generate an answer.", response.text)
        self.assertNotIn('"phase":"complete"', response.text)

    def test_provider_is_direct_openai_and_closes(self):
        async def check():
            client_context = AsyncMock()
            client = client_context.__aenter__.return_value
            with patch(
                "app.dependencies.AsyncOpenAI", return_value=client_context
            ) as factory:
                dependency = get_chat_model(self.config)
                model = await anext(dependency)
                self.assertEqual(model.model_name, self.config.chat_model_name)
                self.assertIs(model.client, client)
                await dependency.aclose()
                factory.assert_called_once_with(api_key="test-key")
                client_context.__aexit__.assert_awaited_once()

        asyncio.run(check())

    def test_blank_keys_are_rejected_by_config_as_http_503(self):
        self.app.dependency_overrides.pop(get_chat_config)
        for key in ("", " \t "):
            with (
                self.subTest(key=repr(key)),
                patch(
                    "app.dependencies.ChatConfig",
                    side_effect=lambda key=key: ChatConfig(
                        _env_file=None, openai_api_key=key
                    ),
                ),
            ):
                response = self.send([user_message("Hello")])
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json(), {"detail": "Chat is not configured"})
        self.assertFalse(self.calls)

    def test_conversion_discards_provider_metadata(self):
        history = [
            {
                "id": "assistant-1",
                "role": "assistant",
                "metadata": {"ignored": True},
                "parts": [
                    {
                        "type": "reasoning",
                        "text": "Earlier reasoning",
                        "providerMetadata": {"openai": {"itemId": "untrusted"}},
                    }
                ],
            },
            {
                "id": "user-1",
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "text": "Hello",
                        "providerMetadata": {"openai": {"itemId": "untrusted"}},
                    }
                ],
            },
        ]
        self.send(history)
        self.assertNotIn("untrusted", repr(self.calls))
        self.assertNotIn("ignored", repr(self.calls))

    def test_history_limits_and_regeneration_are_rejected(self):
        self.assertEqual(
            self.send(
                [user_message("Hello")], trigger="regenerate-message"
            ).status_code,
            422,
        )
        self.assertEqual(
            self.send([user_message("a") for _ in range(101)]).status_code, 422
        )
        self.assertEqual(
            self.send([user_message("a" * 19_000) for _ in range(11)]).status_code, 422
        )
        self.assertFalse(self.calls)


class StreamLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def adapter(self, model):
        return VercelAIAdapter(
            agent=create_book_agent(model),
            run_input=ChatRequest(messages=[user_message("Find evidence")]),
        )

    async def test_search_progress_arrives_before_slow_operations_finish(self):
        embed_release, search_release = asyncio.Event(), asyncio.Event()
        book_id, index_id = uuid4(), uuid4()

        async def embed(query):
            await embed_release.wait()
            return [1.0]

        async def retrieve(function, vector):
            await search_release.wait()
            return []

        embeddings = Mock()
        embeddings.aembed_query = AsyncMock(side_effect=embed)
        with patch("app.services.ai.tools.book_search.Session") as sessions:
            sessions.return_value.__enter__.return_value.get.side_effect = [
                Mock(active_index_id=index_id),
                Mock(id=index_id, book_id=book_id, status=BookIndexStatus.READY),
            ]
            searcher = BookSearch(engine=Mock(), embeddings=embeddings, book_id=book_id)
        model, calls, _ = recorded_model("No evidence found.", query="evidence")
        stream = stream_chat(self.adapter(model), searcher, timeout_seconds=5)

        async def until_phase(phase):
            while True:
                chunk = await asyncio.wait_for(anext(stream), timeout=1)
                event = parse_chunks([chunk])[0]
                if event["type"] == "data-status" and event["data"]["phase"] == phase:
                    return event

        with patch(
            "app.services.ai.tools.book_search.run_in_threadpool", side_effect=retrieve
        ):
            try:
                embedding = await until_phase("embedding-query")
                self.assertFalse(embed_release.is_set())
                embed_release.set()
                searching = await until_phase("searching-book")
                self.assertFalse(search_release.is_set())
                self.assertEqual(
                    embedding["data"]["toolCallId"], searching["data"]["toolCallId"]
                )
                search_release.set()
                events = parse_chunks([chunk async for chunk in stream])
                self.assertTrue(
                    any(
                        e["type"] == "data-status" and e["data"]["phase"] == "complete"
                        for e in events
                    )
                )
                results = [
                    p.content
                    for m in calls[-1]
                    for p in m.parts
                    if isinstance(p, ToolReturnPart)
                ]
                self.assertEqual(
                    results,
                    [
                        {
                            "book_id": str(book_id),
                            "index_id": str(index_id),
                            "passages": [],
                        }
                    ],
                )
            finally:
                embed_release.set()
                search_release.set()
                await stream.aclose()

    async def test_timeout_closes_model_stream(self):
        closed = asyncio.Event()

        async def slow(messages, info):
            try:
                await asyncio.sleep(10)
                yield "late"
            finally:
                closed.set()

        stream = stream_chat(
            self.adapter(FunctionModel(stream_function=slow)),
            Mock(spec=BookSearch),
            timeout_seconds=0.01,
        )
        chunks = [chunk async for chunk in stream]
        self.assertTrue(closed.is_set())
        self.assertIn("timed out", "".join(chunks))
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")

    async def test_cancellation_propagates_and_closes_running_tool(self):
        started, closed = asyncio.Event(), asyncio.Event()

        async def slow_search(query):
            try:
                yield SearchPhase.EMBEDDING_QUERY
                started.set()
                await asyncio.sleep(10)
                yield {}
            finally:
                closed.set()

        searcher = Mock(spec=BookSearch)
        searcher.stream_search.side_effect = slow_search
        model, _, _ = recorded_model(query="evidence")
        stream = stream_chat(self.adapter(model), searcher, timeout_seconds=30)

        async def consume():
            return [chunk async for chunk in stream]

        task = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(closed.is_set())
        await stream.aclose()


class BookSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_uses_captured_index_and_one_based_pages(self):
        book_id, index_id, chunk_id = uuid4(), uuid4(), uuid4()
        embeddings = Mock()
        embeddings.aembed_query = AsyncMock(return_value=[1.0, 0.0])
        with (
            patch("app.services.ai.tools.book_search.Session") as session_factory,
            patch("app.services.ai.tools.book_search.PgVectorStore") as store_factory,
        ):
            session = session_factory.return_value.__enter__.return_value
            book = Mock(active_index_id=index_id)
            session.get.side_effect = [
                book,
                Mock(id=index_id, book_id=book_id, status=BookIndexStatus.READY),
            ]
            search = BookSearch(engine=Mock(), embeddings=embeddings, book_id=book_id)
            book.active_index_id = uuid4()  # A later reindex cannot redirect this run.
            store_factory.return_value.search.return_value = [
                SearchHit(
                    chunk_id=chunk_id,
                    index_id=index_id,
                    text="Evidence",
                    path=("Chapter 1",),
                    pages=(0, 2),
                    distance=0,
                )
            ]
            result = await search.search("query")
            embeddings.aembed_query.assert_awaited_once_with("query")
            store_factory.return_value.search.assert_called_once_with(
                [1.0, 0.0], index_id=index_id
            )
            self.assertEqual(result["index_id"], str(index_id))
            self.assertEqual(result["passages"][0]["pdf_pages"], [1, 3])
            self.assertEqual(session_factory.call_count, 2)
            self.assertEqual(session_factory.return_value.__exit__.call_count, 2)
