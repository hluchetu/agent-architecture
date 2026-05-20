from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Structured response from the LLM — either text or a tool call."""

    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_tool_call(self) -> bool:
        return bool(self.tool_calls)
