# Booker

Booker is a local app for asking questions about PDF books. Upload a book with
bookmarks, and Booker extracts its section structure, chunks and embeds the
text, and lets you chat with it. The agent can search the book or the web for
external information.

## Technology

- **PDF processing:** PyMuPDF for bookmarks and page text.
- **Chunking and AI:** LangChain with OpenAI `text-embedding-3-small` embeddings
  and `gpt-4o-mini` for chat and web search.
- **Storage and UI:** In-memory Chroma for vectors and Chainlit for the chat UI.
- **Development:** Python, uv, unittest, and Ruff.

## Prerequisites

- Python 3.11.4 (pinned in `.python-version`) and [uv](https://docs.astral.sh/uv/).
- An OpenAI API key with access to the configured models.
- A PDF with bookmarks and extractable text. Scanned PDFs without OCR and PDFs
  without bookmarks are not supported yet.

## Install

```sh
uv sync --locked
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Book text is sent to OpenAI for embeddings and
model responses.

## Run

```sh
uv run chainlit run app/chainlit_app.py
```

Open the URL printed by Chainlit and upload a PDF. Indexing may take a while
for a large book. Each chat keeps its conversation and vectors in memory;
Chainlit keeps the uploaded PDF in its session files. A new chat requires
another upload and re-indexes the book.

## Book API

Start PostgreSQL and RustFS, and create the configured S3 bucket before uploading.
The API loads database and storage settings from `.env`; AI credentials are not
needed for these endpoints. For the local Compose database, use:

```dotenv
DATABASE_URL=postgresql+psycopg://admin:mysecret@localhost:5432/booker
S3_BUCKET_NAME=booker-prod
```

Run the API:

```sh
make dev
```

The API creates missing tables on startup. Existing tables are not migrated.
Open `/docs` to try the endpoints:

- `POST /books`: multipart upload with a required `file` field; returns a book
  with HTTP 201. The file must have a `.pdf` extension, readable PDF content, and
  a size of at most 100 MiB. Uploading does not run ingestion.
- `GET /books?offset=0&limit=20`: lists database records newest first, including
  failed attempts. Offset must be non-negative; limit must be between 1 and 100.

Responses contain `id`, `filename`, `status`, and `created_at`. Storage failures
return HTTP 502 with the failed book ID; database failures return HTTP 503.

## Tests and checks

Tests use FastAPI's TestClient with real PostgreSQL and RustFS. Start these
services first and create the `booker-test` database. Configure `TEST_DATABASE_URL`
and `TEST_S3_BUCKET_NAME` in `.env` (defaults are shown in `.env.example`). Pydantic
requires the database and bucket names to be `booker-test`, keeping test cleanup
separate from development data. RustFS credentials and endpoint use the regular
storage settings.

The suite creates the test bucket if missing, uploads generated PDFs, checks
database metadata and stored bytes, and removes only its own records and objects.
The test bucket and tables remain available for future runs. Missing services
cause a test failure rather than a skip. Unit tests are not retained.

```sh
make test
uv run ruff check app tests
uv run ruff format --check app tests
```
