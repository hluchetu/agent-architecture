from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def now_utc() -> datetime:
    return datetime.now(UTC)


class ObservabilityEvent(BaseModel):
    """Structured event emitted by the demo runtime."""

    name: str
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    turn_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=now_utc)


