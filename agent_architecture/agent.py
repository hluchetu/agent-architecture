from __future__ import annotations

from agent_architecture.config import Settings
from agent_architecture.llm import OllamaClient
from agent_architecture.memory import ChatContext
from agent_architecture.observability import EventLog
from agent_architecture.tools import ToolRegistry
from agent_architecture.tools.tool import Tool


class Agent:
    """A self-contained agent with its own context, LLM client, and tools."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        settings: Settings | None = None,
        tools: list[Tool] | None = None,
        event_log: EventLog | None = None,
        max_context_items: int = 20,
    ) -> None:
        self.name = name
        self.settings = settings or Settings()
        self.context = ChatContext(
            event_log=event_log,
            max_context_items=max_context_items,
        )
        self.llm = OllamaClient(self.settings, event_log=event_log)
        self.registry = ToolRegistry()
        for tool in (tools or []):
            self.registry.register(tool)
        self.context.add_message("system", system_prompt)

    async def run(self, task: str) -> str:
        turn_id = self.context.new_turn_id()
        self.context.add_message("user", task, turn_id=turn_id)

        while True:
            response = await self.llm.chat(
                self.context.messages_for_llm(),
                tools=self.registry.schemas(),
            )

            if response.is_tool_call:
                for tool_call in response.tool_calls:
                    fn = tool_call["function"]
                    name = fn["name"]
                    arguments = fn["arguments"]

                    tool_call_item = self.context.add_tool_call(name, arguments, turn_id=turn_id)

                    tool = self.registry.get(name)
                    if tool is None:
                        result = {"error": f"Tool '{name}' not found"}
                    else:
                        result = await tool.execute(arguments)

                    self.context.add_tool_result(tool_call_item.id, result, turn_id=turn_id)
            else:
                self.context.add_message("assistant", response.content, turn_id=turn_id)
                return response.content
