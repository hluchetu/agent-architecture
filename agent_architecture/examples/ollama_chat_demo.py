from __future__ import annotations

import asyncio

from agent_architecture.config import Settings
from agent_architecture.llm import OllamaClient
from agent_architecture.memory import ChatContext
from agent_architecture.observability import EventLog


async def run() -> None:
    settings = Settings()
    event_log = EventLog()
    context = ChatContext(event_log=event_log)

    turn_id = context.new_turn_id()
    context.add_message("system", "You are a concise voice assistant.")
    context.add_message(
        "user",
        "Explain short-term memory in one sentence.",
        turn_id=turn_id,
    )
    context.add_event("user_turn_completed", {"source": "typed_example"}, turn_id=turn_id)
    tool_call = context.add_tool_call(
        "lookup_memory_definition",
        {"topic": "short-term memory"},
        turn_id=turn_id,
    )
    context.add_tool_result(
        tool_call.id,
        {"definition": "Temporary context used during the active session."},
        turn_id=turn_id,
    )

    llm = OllamaClient(settings, event_log=event_log)
    reply = await llm.chat(context.messages_for_llm())

    context.add_message("assistant", reply, turn_id=turn_id)

    print(reply)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
