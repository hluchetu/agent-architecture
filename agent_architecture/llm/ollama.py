from __future__ import annotations

import asyncio
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent_architecture.config import Settings
from agent_architecture.llm.response import LLMResponse
from agent_architecture.observability import EventLog, ObservabilityEvent
from agent_architecture.observability.tokens import TokenTracker


class OllamaClient:
    """Small async client for Ollama's local chat API."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        event_log: EventLog | None = None,
        token_tracker: TokenTracker | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.event_log = event_log
        self.token_tracker = token_tracker
        self._semaphore = asyncio.Semaphore(self.settings.ollama_max_concurrency)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
        reraise=True,
    )
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

        async with self._semaphore:
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

        prompt_tokens: int = data.get("prompt_eval_count", 0)
        completion_tokens: int = data.get("eval_count", 0)

        if self.token_tracker is not None:
            self.token_tracker.add(prompt_tokens, completion_tokens)

        self._emit(
            "llm.response",
            model=self.settings.ollama_model,
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        return LLMResponse(content=content, tool_calls=tool_calls)

    def _emit(self, name: str, **payload: object) -> None:
        if self.event_log is None:
            return

        self.event_log.emit(ObservabilityEvent(name=name, payload=dict(payload)))
