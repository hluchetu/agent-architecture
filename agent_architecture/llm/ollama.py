from __future__ import annotations

import httpx

from agent_architecture.config import Settings


class OllamaClient:
    """Small async client for Ollama's local chat API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    async def chat(self, messages: list[dict[str, str]]) -> str:
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
        return data["message"]["content"]

