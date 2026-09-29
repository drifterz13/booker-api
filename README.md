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

PostgreSQL uses the pinned `pgvector/pgvector:0.8.6-pg18-trixie` image. Start it
with `docker-compose up -d db`. Existing PostgreSQL 18 data uses the same named
volume. Migrations enable the vector extension for fresh databases; the database
user needs permission to create extensions, or an administrator must enable it
before running migrations.

Start PostgreSQL and RustFS, and create the configured S3 bucket before uploading.
The API loads settings from `.env` and requires `OPENAI_API_KEY` for embeddings.
For the local Compose database, use:

```dotenv
DATABASE_URL=postgresql+psycopg://admin:mysecret@localhost:5432/booker
S3_BUCKET_NAME=booker-prod
INGESTION_WORKERS=1
```

For a fresh database, apply migrations before running the API:

```sh
make migrate
make dev
```

The API does not create or alter tables on startup. Run `make migrate` once per
deployment before starting API workers. Alembic reads `DATABASE_URL` from `.env`.

For a database previously initialized with `create_all()`, first verify that its
`book` table matches `migrations/versions/0001_book_baseline.py`: columns, types,
nullability, primary key, unique object key, and `bookstatus` enum labels. After
verification, record the baseline and apply the remaining migration:

```sh
uv run alembic stamp 0001
make migrate
```

Stamping only records a revision; it does not verify or change the schema. Do not
stamp a fresh database or one with a different schema. Future changes use
`make migration message="description"`; review the generated
migration before running `make migrate`. Use `uv run alembic check` to check for
model/schema differences.

Open `/docs` to try the endpoints:

- `POST /books`: multipart upload with a required `file` field; returns a book
  with HTTP 201 after ingestion is ready. The file must have a `.pdf` extension,
  readable PDF content, and a size of at most 100 MiB. Ingestion requires
  bookmarks and extractable text; the request waits for processing to finish.
- `GET /books?offset=0&limit=20`: lists database records newest first, including
  failed attempts. Offset must be non-negative; limit must be between 1 and 100.

Responses contain `id`, `filename`, `status`, `created_at`, and `active_index_id`.
The index ID is null when no version is active. Book status describes the upload
lifecycle, so an ingestion failure leaves the book uploaded and its new index
failed. Unsupported content such as missing bookmarks returns HTTP 422;
embedding or storage failures return HTTP 502, and database failures return
HTTP 503. Ingestion errors include `book_id` and `index_id` for recovery.

PDF validation, extraction, and chunking run in a shared process pool. Synchronous
storage and database operations run in threads; embeddings use native async
requests. `INGESTION_WORKERS` defaults to one worker per API process.

## PostgreSQL vector store

`PgVectorStore(session)` stores embedded passages in a specific `BookIndex`.
`add_chunks(index_id, items)` appends `EmbeddedChunk` objects to a building
version and flushes the rows. The caller commits or rolls back and manages
index status, chunk counts, and activation. Ready and failed versions reject
further writes. Database errors require the caller to roll back the session.

`search(query_vector, index_id=index_id, limit=5)` returns passage text, section
paths, zero-based PDF pages, and cosine distances. The caller selects the ready
index version; the database chooses an exact or HNSW scan. Transaction-local
iterative HNSW scanning supports filtering by index version. The existing
Chainlit ingestion and agent integration will be migrated in the next step.

The vector-store integration test saves chunks in `booker-test` and searches
them from a new session. It checks ranking, the result limit, index filtering,
and returned text and metadata, then removes its own book and index records.
It uses controlled vectors and needs no OpenAI or RustFS connection.

## Book ingestion service

`BookIngestionService.ingest(book_id)` creates a building index, downloads the
uploaded PDF, and runs `extract_book()` and `chunk_book()` as separate tasks in
a caller-owned process pool. `ChunkEmbedder.aembed()` sends sequential batches
through the embedding provider's async client. The embedder must use the model
and dimensions defined in `BookIndex`.

The service saves chunks, marks the index ready, and activates it in one short
database transaction. A failure marks the new index failed and preserves the
previous active version. Temporary files are removed after processing. The
caller supplies and closes the engine, storage, process pool, and embedding
client. Use a process pool with the `spawn` multiprocessing context to avoid
inheriting the API process's database connections.

`BookUploadService` validates PDF readability in the same process pool before
saving the upload. The POST handler then awaits ingestion. FastAPI's lifespan
creates and closes the process pool, HTTP clients, storage, and database engine.

## Tests and checks

Tests use FastAPI's TestClient with real PostgreSQL and RustFS. Start these
services first and create the `booker-test` database. Configure `TEST_DATABASE_URL`
and `TEST_S3_BUCKET_NAME` in `.env` (defaults are shown in `.env.example`). Pydantic
requires the database and bucket names to be `booker-test`, keeping test cleanup
separate from development data. RustFS credentials and endpoint use the regular
storage settings.

API test setup applies Alembic migrations to `booker-test`. If the database still
has an unversioned `book` table, verify and stamp its baseline once using
`DATABASE_URL` set to the test database URL. Shared resource helpers apply
migrations and manage database and storage client cleanup.

The suite creates the test bucket if missing, uploads generated PDFs, checks
database metadata and stored bytes, and removes only its own records and objects.
The test bucket and tables remain available for future runs. Missing services
cause a test failure rather than a skip. Unit tests are not retained.

API and ingestion integration tests use fixed mocked embeddings and need no
OpenAI API key. It processes a tiny generated PDF with real extraction, chunking,
RustFS, and PostgreSQL, and checks index activation, search, and preservation of
the active version after a failed retry.

```sh
make test
uv run ruff check app tests
uv run ruff format --check app tests
```
