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

## Tests and checks

```sh
uv run python -m unittest discover -s tests -v
uv run ruff check app tests
uv run ruff format --check app tests
```
