from __future__ import annotations

from agent_architecture.config import Settings
from agent_architecture.llm.client import LLMClient
from agent_architecture.llm.ollama import OllamaClient
from agent_architecture.observability import EventLog
from agent_architecture.observability.tokens import TokenTracker


def create_llm_client(
    settings: Settings | None = None,
    event_log: EventLog | None = None,
    token_tracker: TokenTracker | None = None,
) -> LLMClient:
    settings = settings or Settings()

    if settings.inference_engine == "ollama":
        return OllamaClient(settings, event_log=event_log, token_tracker=token_tracker)

    raise ValueError(f"Unsupported inference engine: {settings.inference_engine!r}")
