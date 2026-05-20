from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class SessionStore:
    """Persists ChatContext sessions to SQLite for episodic memory."""

    def __init__(self, db_path: str | Path = "sessions.db") -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                items       TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def save(self, session_id: str, items: list[dict[str, Any]]) -> None:
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO sessions (session_id, created_at, updated_at, items)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                items      = excluded.items
            """,
            (session_id, now, now, json.dumps(items)),
        )
        self._conn.commit()

    def load(self, session_id: str) -> list[dict[str, Any]] | None:
        row = self._conn.execute(
            "SELECT items FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def delete(self, session_id: str) -> None:
        self._conn.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        self._conn.commit()
