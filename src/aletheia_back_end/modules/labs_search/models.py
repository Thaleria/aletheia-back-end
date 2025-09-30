"""Pydantic models."""

from typing import Any, Literal

from pydantic import BaseModel


class Message(BaseModel):
    """Model for individual chat messages."""

    content: str
    role: Literal["user", "assistant"]


class Overrides(BaseModel):
    """Model for context overrides."""

    top: int
    minimum_search_score: int
    retrieval_mode: str
    semantic_captions: bool
    suggest_followup_questions: bool
    use_oid_security_filter: bool
    use_groups_security_filter: bool
    gpt4v_input: str


class Context(BaseModel):
    """Model for chat context."""

    overrides: Overrides


class ChatIn(BaseModel):
    """Input model for Chat."""

    messages: list[
        Message
    ]  # TODO: List? We'll probably just use one at a time, so it'll probably just be Message
    context: Context
    session_state: Any | None = None
