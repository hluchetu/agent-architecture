from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class Tool(ABC):
    """Base class for all agent tools."""

    name: str
    description: str
    parameters: dict[str, Any]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        ...

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
