# Booker contributor notes

Keep changes small, clear, and testable. Keep PDF ingestion, application logic,
AI tools, storage, and the chat API decoupled; avoid putting UI concerns in
the core pipeline. Add or update focused tests when behavior changes.

## Structure

- `app/models/`: book, content, chunk, and search data models.
- `app/services/extractor/`, `chunker/`, `ingest/`: PDF extraction, chunking,
  and indexing.
- `app/services/ai/`: Pydantic AI chat, stream handling, and tools.
- `app/db/vector_store/`: PostgreSQL vector storage.
- `app/services/storage/`: PDF object storage.
- `app/routers/`, `app/schemas/`: FastAPI endpoints and request/response schemas.
- `tests/`: unittest suite.

## Commands

```sh
uv sync --locked
uv run fastapi dev
uv run python -m unittest discover -s tests -v
uv run ruff check app tests
uv run ruff format --check app tests
```

Use `uv lock` when dependencies change; do not update the lockfile for unrelated
code changes.
