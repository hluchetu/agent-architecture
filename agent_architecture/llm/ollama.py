from __future__ import annotations

import httpx

from agent_architecture.config import Settings
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

    async def chat(self, messages: list[dict[str, str]]) -> str:
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
                json={
                    "model": self.settings.ollama_model,
                    "messages": messages,
                    "stream": False,
                },
            )

        response.raise_for_status()
        data = response.json()
        reply: str = data["message"]["content"]
        self._emit(
            "llm.response",
            model=self.settings.ollama_model,
            content=reply,
        )
        return reply

    def _emit(self, name: str, **payload: object) -> None:
        if self.event_log is None:
            return

        self.event_log.emit(ObservabilityEvent(name=name, payload=dict(payload)))
