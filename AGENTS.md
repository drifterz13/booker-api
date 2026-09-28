# Booker contributor notes

Keep changes small, clear, and testable. Keep PDF ingestion, application logic,
AI tools, storage, and the Chainlit UI decoupled; avoid putting UI concerns in
the core pipeline. Add or update focused tests when behavior changes.

## Structure

- `src/booker/model/`: book, content, chunk, and search data models.
- `src/booker/extractor/`, `chunk/`, `ingest/`: PDF extraction, chunking, and
  indexing.
- `src/booker/ai/`: agent, model setup, and tools.
- `src/booker/store/`: Chroma storage.
- `src/booker/application.py`: session orchestration; `chainlit_app.py`: UI.
- `tests/`: unittest suite.

## Commands

```sh
uv sync --locked
uv run chainlit run src/booker/chainlit_app.py
uv run python -m unittest discover -s tests -v
uv run ruff check src tests
uv run ruff format --check src tests
```

Use `uv lock` when dependencies change; do not update the lockfile for unrelated
code changes.
