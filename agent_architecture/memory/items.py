from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


Role = Literal["system", "user", "assistant"]
ContextItemKind = Literal[
    "message",
    "tool_call",
    "tool_result",
    "event",
    "summary",
]


def new_id() -> str:
    return str(uuid4())


def now_utc() -> datetime:
    return datetime.now(UTC)


class ContextItem(BaseModel):
    """Base item stored in short-term memory.

    session_id groups everything for one active conversation.
    turn_id groups all items created during one user turn.
    id identifies this exact memory item.
    """

    kind: ContextItemKind
    id: str = Field(default_factory=new_id)
    session_id: str
    turn_id: str | None = None
    created_at: datetime = Field(default_factory=now_utc)


class MessageItem(ContextItem):
    """A normal chat message that can be sent to an LLM."""

    kind: Literal["message"] = "message"
    role: Role
    content: str


class ToolCallItem(ContextItem):
    """A tool call the agent decided to make."""

    kind: Literal["tool_call"] = "tool_call"
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultItem(ContextItem):
    """The result returned by a tool call."""

    kind: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class EventItem(ContextItem):
    """A session event that may matter for reasoning or debugging."""

    kind: Literal["event"] = "event"
    name: str
    data: dict[str, Any] = Field(default_factory=dict)


class SummaryItem(ContextItem):
    """Compressed context from older turns."""

    kind: Literal["summary"] = "summary"
    content: str

