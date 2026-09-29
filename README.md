# Booker

Booker ingests PDF books, extracts bookmarked sections, and indexes their text
for search. It provides a FastAPI API for uploads, book management, ingestion,
and streaming book chat. PDFs need bookmarks and extractable text.

## Stack

- Python, FastAPI, SQLModel, and Alembic
- Pydantic AI for chat; OpenAI for models and embeddings
- PyMuPDF and LangChain text splitting
- PostgreSQL with pgvector and RustFS for PDF storage
- uv and Ruff

## Installation

Install Python 3.12, uv, and Docker with Compose, then run:

```sh
uv sync --locked
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. The example includes local RustFS credentials;
the API defaults to the Compose PostgreSQL database and the `booker-prod` bucket.

```sh
docker compose up -d db rustfs
```

Open the RustFS console at http://localhost:9001, sign in with the
`RUSTFS_ACCESS_KEY` and `RUSTFS_SECRET_KEY` from `.env`, and create the
`booker-prod` bucket.

## Run

Apply database migrations, then start the API:

```sh
make migrate
make dev
```

Open http://localhost:8000/docs to use the API.

The Docker image starts the same FastAPI application on port 8000.

## Chat

After uploading and ingesting a book, connect React's Vercel AI SDK `useChat`
transport to `POST /books/{book_id}/chat`. The book must have a ready active index.

```sh
curl -N http://localhost:8000/books/BOOK_ID/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"id":"user-1","role":"user","parts":[{"type":"text","text":"What is this book about?"}]}]}'
```

The endpoint streams text, `search_book` tool calls/results, and transient
`data-status` progress events. Handle progress through `useChat`'s `onData`
callback. Send the client conversation history on every request; the backend
does not persist conversations. Model calls go directly to OpenAI without a
gateway. Configure `CHAT_MODEL_NAME` and `CHAT_TIMEOUT_SECONDS` in `.env`.
