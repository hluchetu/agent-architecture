from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Base class for all agent tools."""

    name: str
    description: str
    parameters: dict[str, Any]

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
