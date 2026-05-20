from __future__ import annotations

from typing import Any


class WorkingMemory:
    """Key-value scratchpad the agent reads and writes mid-session.

    Tracks active task state across multiple steps within a session.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def dump(self) -> dict[str, Any]:
        return dict(self._store)
