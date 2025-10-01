"""Data models for Chat."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Model for individual chat messages."""

    content: str
    role: Literal["user", "assistant"]


class Overrides(BaseModel):
    """Model for context overrides."""

    top: int = Field(gt=0, default=3)
    temperature: float = Field(gt=0, le=1, default=0.5)


class Context(BaseModel):
    """Model for chat context."""

    overrides: Overrides


class ChatIn(BaseModel):
    """Input model for Chat."""

    messages: list[
        Message
    ]  # List? We'll probably just use one at a time, so it'll probably just be Message
    context: Context
    party: Any | int |None = None
    session_state: Any | None = None


class ChatResponse(BaseModel):
    """Chat response model."""

    content: str
    role: str


class ChatOut(BaseModel):
    """Output model for Chat."""

    delta: ChatResponse
