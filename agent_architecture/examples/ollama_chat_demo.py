from __future__ import annotations

import asyncio
import sys
from uuid import uuid4

from agent_architecture.config import Settings
from agent_architecture.observability import EventLog, TokenTracker, configure_logging
from agent_architecture.router import Router
from agent_architecture.runner import run_with_graceful_shutdown


async def run(session_id: str, task: str) -> None:
    settings = Settings()
    event_log = EventLog()
    token_tracker = TokenTracker()

    print(f"[session] {session_id}")
    print(f"[task] {task}")

    router = Router(settings=settings, event_log=event_log, token_tracker=token_tracker)
    final = await router.run(task)

    print("\n--- Final Answer ---")
    print(final)
    print(f"\n--- Token Usage ---")
    print(token_tracker.dump())


def main() -> None:
    configure_logging()
    session_id = str(uuid4())
    task = (
        " ".join(sys.argv[1:])
        or "Explain the difference between short-term, episodic, semantic, and working memory."
    )
    asyncio.run(run_with_graceful_shutdown(run(session_id, task)))


if __name__ == "__main__":
    main()
