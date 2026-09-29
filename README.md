# Booker

Booker ingests PDF books, extracts bookmarked sections, and indexes their text
for search. It provides a FastAPI API for managing books and a Chainlit UI for
asking questions about them. PDFs need bookmarks and extractable text.

## Stack

- Python, FastAPI, SQLModel, and Alembic
- PyMuPDF, LangChain, and OpenAI
- PostgreSQL with pgvector, RustFS for PDF storage, and Chroma for Chainlit search
- Chainlit, uv, and Ruff

## Installation

Install Python 3.11.4, uv, and Docker with Compose, then run:

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

To start the chat UI:

```sh
make chainlit
```

Open the URL printed by Chainlit and upload a PDF.
