from __future__ import annotations

import asyncio

from agent_architecture.config import Settings
from agent_architecture.llm import OllamaClient
from agent_architecture.memory import ChatContext, SessionStore
from agent_architecture.observability import EventLog
from agent_architecture.tools import MemoryLookupTool, ToolRegistry

SESSION_ID = "demo-session-001"


async def agent_loop(
    context: ChatContext,
    llm: OllamaClient,
    registry: ToolRegistry,
    turn_id: str,
) -> str:
    while True:
        response = await llm.chat(context.messages_for_llm(), tools=registry.schemas())

        if response.is_tool_call:
            for tool_call in response.tool_calls:
                fn = tool_call["function"]
                name = fn["name"]
                arguments = fn["arguments"]

                tool_call_item = context.add_tool_call(name, arguments, turn_id=turn_id)

                tool = registry.get(name)
                if tool is None:
                    result = {"error": f"Tool '{name}' not found"}
                else:
                    result = await tool.execute(arguments)

                context.add_tool_result(tool_call_item.id, result, turn_id=turn_id)
        else:
            context.add_message("assistant", response.content, turn_id=turn_id)
            return response.content


async def run() -> None:
    settings = Settings()
    event_log = EventLog()
    store = SessionStore()

    registry = ToolRegistry()
    registry.register(MemoryLookupTool())

    context = ChatContext(
        session_id=SESSION_ID,
        event_log=event_log,
        max_context_items=settings.max_context_items,
    )

    resumed = context.resume(store)
    if resumed:
        print(f"[session] resumed {SESSION_ID} with {len(context.items)} items")
    else:
        print(f"[session] starting fresh session {SESSION_ID}")
        context.add_message("system", "You are a concise assistant that uses tools to look up definitions.")

    turn_id = context.new_turn_id()
    context.add_message(
        "user",
        "What is short-term memory?",
        turn_id=turn_id,
    )

    llm = OllamaClient(settings, event_log=event_log)
    reply = await agent_loop(context, llm, registry, turn_id)

    store.save(SESSION_ID, context.dump())
    print(f"[session] saved {SESSION_ID} with {len(context.items)} items")

    print(reply)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
