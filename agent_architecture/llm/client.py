from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    """Protocol that any LLM client must implement."""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        ...
