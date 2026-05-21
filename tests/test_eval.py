from __future__ import annotations

import pytest

from agent_architecture.eval import Evaluator
from agent_architecture.llm.response import LLMResponse
from agent_architecture.memory import ChatContext
from agent_architecture.tools import MemoryLookupTool, ToolRegistry


class MockLLMClient:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = iter(responses)

    async def chat(self, messages, tools=None) -> LLMResponse:
        return next(self.responses)


class MockEvaluator:
    """Evaluator that scores by keyword matching — no LLM needed."""

    async def evaluate(self, question: str, answer: str, rubric: str) -> int:
        keywords = rubric.lower().split(",")
        answer_lower = answer.lower()
        matches = sum(1 for kw in keywords if kw.strip() in answer_lower)
        return min(5, max(1, matches + 1))


async def test_evaluator_scores_correct_answer():
    evaluator = MockEvaluator()
    score = await evaluator.evaluate(
        question="What is short-term memory?",
        answer="Short-term memory is temporary context used during the active session.",
        rubric="temporary, session, context",
    )
    assert score >= 3


async def test_evaluator_scores_wrong_answer_low():
    evaluator = MockEvaluator()
    score = await evaluator.evaluate(
        question="What is short-term memory?",
        answer="I don't know.",
        rubric="temporary, session, context",
    )
    assert score <= 2


async def test_evaluator_scores_partial_answer():
    evaluator = MockEvaluator()
    score = await evaluator.evaluate(
        question="What is short-term memory?",
        answer="Short-term memory is temporary.",
        rubric="temporary, session, context",
    )
    assert 2 <= score <= 4


async def test_full_agent_answer_scored():
    context = ChatContext()
    registry = ToolRegistry()
    registry.register(MemoryLookupTool())

    mock_llm = MockLLMClient([
        LLMResponse(
            content="",
            tool_calls=[{
                "function": {
                    "name": "lookup_memory_definition",
                    "arguments": {"topic": "working memory"},
                }
            }],
        ),
        LLMResponse(
            content="Working memory is active task state tracked mid-session.",
            tool_calls=[],
        ),
    ])

    turn_id = context.new_turn_id()
    await context.add_message("user", "What is working memory?", turn_id=turn_id)

    while True:
        response = await mock_llm.chat(context.messages_for_llm(), tools=registry.schemas())
        if response.is_tool_call:
            for tc in response.tool_calls:
                fn = tc["function"]
                tool_call_item = await context.add_tool_call(fn["name"], fn["arguments"], turn_id=turn_id)
                tool = registry.get(fn["name"])
                result = await tool.execute(fn["arguments"]) if tool else {"error": "not found"}
                await context.add_tool_result(tool_call_item.id, result, turn_id=turn_id)
        else:
            reply = response.content
            break

    evaluator = MockEvaluator()
    score = await evaluator.evaluate(
        question="What is working memory?",
        answer=reply,
        rubric="active, task, session",
    )
    assert score >= 3
