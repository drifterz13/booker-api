from pathlib import Path

import chainlit as cl
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from app.core import Config
from app.core.chainlit import Application, Session
from app.db.vector_store import ChromaStore
from app.services.ai import create_openai_model

EMBEDDING_MODEL = "text-embedding-3-small"

load_dotenv()


@cl.on_chat_start
async def on_chat_start() -> None:
    files = await cl.AskFileMessage(
        content="Upload a book (.pdf) to begin.",
        accept=["application/pdf"],
        max_size_mb=100,
        max_files=1,
        timeout=300,
    ).send()
    if not files:
        await cl.Message(
            content="Upload timed out. Start a new chat to try again."
        ).send()
        return

    uploaded = files[0]
    if Path(uploaded.name).suffix.lower() != ".pdf":
        await cl.Message(content="Please upload a PDF file.").send()
        return

    status = await cl.Message(content=f"Indexing {uploaded.name}...").send()

    store: ChromaStore | None = None
    conf = Config()
    try:
        store = ChromaStore(config=conf)
        application = Application(
            store=store,
            embeddings=OpenAIEmbeddings(model=EMBEDDING_MODEL),
            model=create_openai_model(),
        )
        session = await cl.make_async(application.start)(Path(uploaded.path))
    except Exception as exc:  # noqa: BLE001 - show indexing failures in the UI
        if store is not None:
            store.close()
        status.content = f"Could not index {uploaded.name}: {exc}"
        await status.update()
        return

    cl.user_session.set("session", session)
    cl.user_session.set("store", store)
    status.content = f"Book: {uploaded.name} is ready. Ask about the book!"
    await status.update()
    await cl.Pdf(path=str(session.source), name=uploaded.name, display="side").send(
        for_id=status.id
    )


@cl.on_chat_end
async def on_chat_end() -> None:
    store: ChromaStore | None = cl.user_session.get("store")
    if store is not None:
        store.close()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    session: Session | None = cl.user_session.get("session")
    if session is None:
        await cl.Message(content="Upload a book in a new chat first.").send()
        return

    answer = await cl.Message(content="Working on your answer...").send()
    started = False
    try:
        async for token in session.stream(message.content):
            if not started:
                answer.content = ""
                await answer.update()
                started = True
            await answer.stream_token(token)
    except Exception as exc:  # noqa: BLE001 - show agent failures in the UI
        answer.content = f"Could not answer: {exc}"
        await answer.update()
        return

    if not started:
        answer.content = "No answer was generated."
    await answer.update()
