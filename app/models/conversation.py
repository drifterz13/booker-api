from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

from .book import Book  # noqa: F401 - register the referenced book table


class Conversation(SQLModel, table=True):
    """A public conversation about one book."""

    __table_args__ = (
        Index("ix_conversation_book_updated_at", "book_id", "updated_at"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    book_id: UUID = Field(foreign_key="book.id", ondelete="CASCADE")
    title: str | None = Field(default=None)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=DateTime(timezone=True)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=DateTime(timezone=True)
    )


class ConversationRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ConversationMessage(SQLModel, table=True):
    """One visible message, with optional Pydantic AI messages for replay.

    UI data is stored separately because adapter conversion drops emitted
    data-citations parts. An assistant row can hold the agent messages from
    its completed run, including the user prompt and tool calls.
    """

    __table_args__ = (
        UniqueConstraint(
            "conversation_id", "position", name="uq_conversationmessage_position"
        ),
        CheckConstraint("position >= 0", name="ck_conversationmessage_position"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    conversation_id: UUID = Field(foreign_key="conversation.id", ondelete="CASCADE")
    position: int = Field(ge=0)
    role: ConversationRole
    ui_message: dict[str, Any] = Field(sa_type=JSONB)
    model_messages: list[dict[str, Any]] | None = Field(default=None, sa_type=JSONB)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_type=DateTime(timezone=True)
    )
