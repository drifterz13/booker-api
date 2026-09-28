from collections.abc import AsyncIterator
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from ..services.ingest import reindex
from ..services.ai import BookAgent, build_book_agent, WebSearcher
from ..db.vector_store import ChromaStore


class Session:
    def __init__(self, source: Path, agent: BookAgent) -> None:
        self.source = source
        self._agent = agent
        self._messages: list[dict[str, str]] = []

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        messages = [*self._messages, {"role": "user", "content": prompt}]
        parts: list[str] = []
        async for token in self._agent.stream(messages):
            parts.append(token)
            yield token

        if parts:
            self._messages.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "".join(parts)},
                ]
            )


class Application:
    def __init__(
        self,
        *,
        store: ChromaStore,
        embeddings: Embeddings,
        model: BaseChatModel,
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._model = model

    def start(self, upload: Path) -> Session:
        reindex(
            upload,
            store=self._store,
            embeddings=self._embeddings,
        )
        agent = build_book_agent(
            model=self._model,
            source=upload,
            store=self._store,
            embeddings=self._embeddings,
            web_searcher=WebSearcher(self._model),
        )
        return Session(upload, agent)
