from __future__ import annotations

import asyncio
import signal
from collections.abc import Coroutine
from typing import Any

import structlog

logger = structlog.get_logger()


async def run_with_graceful_shutdown(coro: Coroutine[Any, Any, Any]) -> None:
    """Run a coroutine and handle SIGINT/SIGTERM with a clean exit.

    On signal: cancels the task, waits for it to finish, then returns.
    The caller's asyncio.run() exits normally instead of raising KeyboardInterrupt.
    """
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(coro)

    def _handle_signal(sig: signal.Signals) -> None:
        logger.info("shutdown.signal_received", signal=sig.name)
        task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal, sig)

    try:
        await task
    except asyncio.CancelledError:
        logger.info("shutdown.complete")
