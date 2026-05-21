from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for local agent architecture examples.

    Values can be overridden with environment variables:

    AGENT_ARCH_OLLAMA_MODEL=gemma3:4b
    AGENT_ARCH_OLLAMA_BASE_URL=http://localhost:11434
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENT_ARCH_",
        env_file=".env",
        extra="ignore",
    )

    inference_engine: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_timeout_seconds: float = 120.0
    ollama_max_concurrency: int = Field(default=3, ge=1)
    max_context_items: int = Field(default=20, ge=1)

