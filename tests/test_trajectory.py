from __future__ import annotations

from typing import Any

import pytest

from agent_architecture.llm.response import LLMResponse
from agent_architecture.memory import ChatContext
from agent_architecture.memory.items import ToolCallItem, ToolResultItem
from agent_architecture.tools import MemoryLookupTool, ToolRegistry


class MockLLMClient:
    """LLM client that returns pre-configured responses for testing."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        return next(self.responses)


async def run_agent_loop(
    context: ChatContext,
    llm: MockLLMClient,
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

                tool_call_item = await context.add_tool_call(name, arguments, turn_id=turn_id)

                tool = registry.get(name)
                if tool is None:
                    result = {"error": f"Tool '{name}' not found"}
                else:
                    result = await tool.execute(arguments)

                await context.add_tool_result(tool_call_item.id, result, turn_id=turn_id)
        else:
            await context.add_message("assistant", response.content, turn_id=turn_id)
            return response.content


async def test_agent_calls_tool_then_answers():
    context = ChatContext()
    registry = ToolRegistry()
    registry.register(MemoryLookupTool())

    mock = MockLLMClient([
        LLMResponse(
            content="",
            tool_calls=[{
                "function": {
                    "name": "lookup_memory_definition",
                    "arguments": {"topic": "episodic memory"},
                }
            }],
        ),
        LLMResponse(
            content="Episodic memory stores past conversations.",
            tool_calls=[],
        ),
    ])

    turn_id = context.new_turn_id()
    await context.add_message("user", "What is episodic memory?", turn_id=turn_id)
    reply = await run_agent_loop(context, mock, registry, turn_id)

    assert reply == "Episodic memory stores past conversations."
    assert len(mock.calls) == 2


async def test_tool_result_stored_in_context():
    context = ChatContext()
    registry = ToolRegistry()
    registry.register(MemoryLookupTool())

    mock = MockLLMClient([
        LLMResponse(
            content="",
            tool_calls=[{
                "function": {
                    "name": "lookup_memory_definition",
                    "arguments": {"topic": "short-term memory"},
                }
            }],
        ),
        LLMResponse(content="Here is the answer.", tool_calls=[]),
    ])

    turn_id = context.new_turn_id()
    await context.add_message("user", "What is short-term memory?", turn_id=turn_id)
    await run_agent_loop(context, mock, registry, turn_id)

    tool_calls = [i for i in context.items if isinstance(i, ToolCallItem)]
    tool_results = [i for i in context.items if isinstance(i, ToolResultItem)]

    assert len(tool_calls) == 1
    assert len(tool_results) == 1
    assert tool_calls[0].name == "lookup_memory_definition"


async def test_agent_handles_missing_tool_gracefully():
    context = ChatContext()
    registry = ToolRegistry()

    mock = MockLLMClient([
        LLMResponse(
            content="",
            tool_calls=[{
                "function": {
                    "name": "nonexistent_tool",
                    "arguments": {},
                }
            }],
        ),
        LLMResponse(content="I could not find that tool.", tool_calls=[]),
    ])

    turn_id = context.new_turn_id()
    await context.add_message("user", "Use a tool", turn_id=turn_id)
    reply = await run_agent_loop(context, mock, registry, turn_id)

    tool_results = [i for i in context.items if isinstance(i, ToolResultItem)]
    assert "error" in tool_results[0].result


async def test_agent_answers_directly_without_tool():
    context = ChatContext()
    registry = ToolRegistry()

    mock = MockLLMClient([
        LLMResponse(content="Hello! How can I help?", tool_calls=[]),
    ])

    turn_id = context.new_turn_id()
    await context.add_message("user", "Say hello", turn_id=turn_id)
    reply = await run_agent_loop(context, mock, registry, turn_id)

    assert reply == "Hello! How can I help?"
    assert len(mock.calls) == 1


async def test_multiple_tool_calls_in_sequence():
    context = ChatContext()
    registry = ToolRegistry()
    registry.register(MemoryLookupTool())

    mock = MockLLMClient([
        LLMResponse(
            content="",
            tool_calls=[{
                "function": {
                    "name": "lookup_memory_definition",
                    "arguments": {"topic": "short-term memory"},
                }
            }],
        ),
        LLMResponse(
            content="",
            tool_calls=[{
                "function": {
                    "name": "lookup_memory_definition",
                    "arguments": {"topic": "long-term memory"},
                }
            }],
        ),
        LLMResponse(content="Here is the comparison.", tool_calls=[]),
    ])

    turn_id = context.new_turn_id()
    await context.add_message("user", "Compare short and long term memory", turn_id=turn_id)
    reply = await run_agent_loop(context, mock, registry, turn_id)

    tool_calls = [i for i in context.items if isinstance(i, ToolCallItem)]
    assert len(tool_calls) == 2
    assert reply == "Here is the comparison."
