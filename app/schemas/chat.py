from typing import Self
from uuid import uuid4

from pydantic import Field, model_validator
from pydantic_ai.ui.vercel_ai.request_types import (
    DataUIPart,
    ReasoningUIPart,
    StepStartUIPart,
    SubmitMessage,
    TextUIPart,
    ToolInputAvailablePart,
    ToolInputStreamingPart,
    ToolOutputAvailablePart,
    ToolOutputErrorPart,
    UIMessage,
)

from .citation import CitationData


class ChatRequest(SubmitMessage, extra="ignore"):
    """SDK wire models with Booker's text-only chat policy."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    messages: list[UIMessage] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_history(self) -> Self:
        for message in self.messages:
            if message.role not in {"user", "assistant"}:
                raise ValueError("Only user and assistant messages are supported")
            message.metadata = None
            for part in message.parts:
                if (
                    isinstance(part, TextUIPart)
                    or message.role == "assistant"
                    and isinstance(part, ReasoningUIPart)
                ):
                    part.provider_metadata = None
                elif message.role == "assistant" and isinstance(part, StepStartUIPart):
                    pass
                elif message.role == "assistant" and isinstance(part, DataUIPart):
                    if part.type != "data-citations":
                        raise ValueError("Only citation data history is supported")
                    # Accepted for UI round-tripping, never trusted as current evidence.
                    part.data = CitationData.model_validate(part.data).model_dump(
                        mode="json"
                    )
                elif message.role == "assistant" and isinstance(
                    part,
                    ToolInputStreamingPart
                    | ToolInputAvailablePart
                    | ToolOutputAvailablePart
                    | ToolOutputErrorPart,
                ):
                    if part.type != "tool-search_book" or part.approval is not None:
                        raise ValueError("Only book-search tool history is supported")
                    part.call_provider_metadata = None
                    part.provider_executed = None
                else:
                    raise ValueError("Unsupported chat message part")
        message = self.messages[-1]
        if message.role != "user":
            raise ValueError("The last message must be a user message")
        text = "".join(part.text for part in message.parts)
        if not text.strip() or len(text) > 20_000:
            raise ValueError("The user message requires 1 to 20000 characters")
        if len(self.model_dump_json()) > 200_000:
            raise ValueError("Chat history exceeds 200000 characters")
        return self
