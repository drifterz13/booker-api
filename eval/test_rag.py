"""Run explicitly with: uv run --group eval deepeval test run eval/test_rag.py."""

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from deepeval import assert_test
from deepeval.metrics import ContextualRecallMetric, FaithfulnessMetric, GEval
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI
from pydantic import Field, TypeAdapter
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.core.config import ChatConfig, DatabaseConfig, Settings
from app.db.database import create_db_engine
from app.models.book_index import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from app.services.ai.tools.book_search import BookSearch
from eval.support import run_question, write_artifact


class EvalConfig(Settings):
    eval_book_ids_file: Path = Path("eval/books.local.json")
    eval_judge_model: str = "gpt-4o"
    eval_threshold: float = Field(default=0.7, gt=0, le=1)
    eval_results_dir: Path = Path("eval/results")


DATASET_PATH = Path(__file__).with_name("rag-evaluation-draft.json")
GOLDENS = json.loads(DATASET_PATH.read_text())
CASE_IDS = [golden["additional_metadata"]["case_id"] for golden in GOLDENS]


@pytest.fixture(scope="session")
def configuration():
    config = EvalConfig()
    chat = ChatConfig()  # pyright: ignore[reportCallIssue]
    book_ids = TypeAdapter(dict[str, UUID]).validate_json(
        config.eval_book_ids_file.read_text()
    )
    # Booker reads .env via Pydantic; the DeepEval judge reads the environment.
    os.environ.setdefault("OPENAI_API_KEY", chat.openai_api_key.get_secret_value())
    return config, chat, book_ids


@pytest.fixture(scope="session")
def engine():
    engine = create_db_engine(DatabaseConfig())
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope="session")
def results_dir(configuration):
    config, _, _ = configuration
    path = config.eval_results_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path.mkdir(parents=True)
    return path


async def generate(golden, engine, chat, book_id):
    async with (
        AsyncOpenAI(api_key=chat.openai_api_key.get_secret_value()) as client,
        httpx.AsyncClient() as async_http,
    ):
        with httpx.Client() as sync_http:
            embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                dimensions=EMBEDDING_DIMENSIONS,
                api_key=chat.openai_api_key,
                http_client=sync_http,
                http_async_client=async_http,
                max_retries=2
            )
            searcher = BookSearch(engine=engine, embeddings=embeddings, book_id=book_id)
            model = OpenAIChatModel(
                chat.chat_model_name, provider=OpenAIProvider(openai_client=client)
            )
            return await run_question(
                searcher,
                model,
                golden["input"],
                timeout_seconds=chat.chat_timeout_seconds,
            )


def make_metrics(config, behavior):
    steps = [
        (
            "Compare actual_output with expected_output for the question in input. "
            "Accept equivalent wording and ignore Markdown citation formatting."
        ),
        (
            "Penalize factual contradictions and missing essential parts of the "
            "expected answer. Extra details must be supported by retrieval_context."
        ),
    ]
    if behavior == "abstain":
        steps.append(
            "The expected behavior is abstention. The actual_output must clearly "
            "acknowledge missing evidence for the requested fact and must not "
            "invent a number or assert an unsupported answer. Supported background "
            "information is acceptable."
        )
    metrics: list[BaseMetric] = [
        GEval(
            name="Answer correctness" if behavior == "answer" else "Abstention",
            evaluation_steps=steps,
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
                SingleTurnParams.RETRIEVAL_CONTEXT,
            ],
            model=config.eval_judge_model,
            threshold=config.eval_threshold,
        )
    ]
    if behavior == "answer":
        metrics.extend(
            [
                FaithfulnessMetric(
                    model=config.eval_judge_model, threshold=config.eval_threshold
                ),
                ContextualRecallMetric(
                    model=config.eval_judge_model, threshold=config.eval_threshold
                ),
            ]
        )
    return metrics


@pytest.mark.parametrize("golden", GOLDENS, ids=CASE_IDS)
def test_rag(golden, configuration, engine, results_dir):
    config, chat, book_ids = configuration
    metadata = golden["additional_metadata"]
    observation = asyncio.run(
        generate(golden, engine, chat, book_ids[metadata["book_key"]])
    )
    artifact = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "chat_model": chat.chat_model_name,
        "judge_model": config.eval_judge_model,
        "threshold": config.eval_threshold,
        "golden": golden,
        **observation,
    }
    metrics = make_metrics(config, metadata["expected_behavior"])
    try:
        assert observation["searches"], "The agent did not search the book"
        if metadata["expected_behavior"] == "answer":
            assert observation["retrieval_context"], "Search returned no evidence"
        assert_test(
            LLMTestCase(
                input=golden["input"],
                actual_output=observation["actual_output"],
                expected_output=golden["expected_output"],
                context=golden["context"],
                retrieval_context=observation["retrieval_context"],
                metadata=metadata,
            ),
            metrics=metrics,
            # Async execution copies metrics, leaving these instances unscored.
            run_async=False,
        )
    finally:
        artifact["metrics"] = [
            {
                "name": metric.__name__,
                "score": metric.score,
                "reason": metric.reason,
                "error": metric.error,
            }
            for metric in metrics
        ]
        path = results_dir / f"{metadata['case_id']}.json"
        write_artifact(path, artifact)
        print(f"Evaluation artifact: {path}")
