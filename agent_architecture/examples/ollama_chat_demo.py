from __future__ import annotations

import asyncio
import sys
from uuid import uuid4

from agent_architecture.config import Settings
from agent_architecture.observability import EventLog
from agent_architecture.router import Router


async def run(session_id: str, task: str) -> None:
    settings = Settings()
    event_log = EventLog()

    print(f"[session] {session_id}")
    print(f"[task] {task}")

    router = Router(settings=settings, event_log=event_log)
    final = await router.run(task)

    print("\n--- Final Answer ---")
    print(final)


def main() -> None:
    session_id = str(uuid4())
    task = (
        " ".join(sys.argv[1:])
        or "Explain the difference between short-term, episodic, semantic, and working memory."
    )
    asyncio.run(run(session_id, task))


if __name__ == "__main__":
    main()
