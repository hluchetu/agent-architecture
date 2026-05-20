from __future__ import annotations

from typing import Any

import httpx

from agent_architecture.config import Settings
from agent_architecture.llm.response import LLMResponse
from agent_architecture.observability import EventLog, ObservabilityEvent


class OllamaClient:
    """Small async client for Ollama's local chat API."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        event_log: EventLog | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.event_log = event_log

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.settings.ollama_model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        self._emit(
            "llm.request",
            model=self.settings.ollama_model,
            message_count=len(messages),
        )

        async with httpx.AsyncClient(
            timeout=self.settings.ollama_timeout_seconds
        ) as client:
            response = await client.post(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/chat",
                json=payload,
            )

        response.raise_for_status()
        data = response.json()
        message = data["message"]

        content: str = message.get("content", "")
        tool_calls: list[dict[str, Any]] = message.get("tool_calls") or []

        self._emit(
            "llm.response",
            model=self.settings.ollama_model,
            content=content,
            tool_calls=tool_calls,
        )

        return LLMResponse(content=content, tool_calls=tool_calls)

    def _emit(self, name: str, **payload: object) -> None:
        if self.event_log is None:
            return

        self.event_log.emit(ObservabilityEvent(name=name, payload=dict(payload)))
