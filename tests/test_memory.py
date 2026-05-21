from __future__ import annotations

import pytest

from agent_architecture.memory import ChatContext, SessionStore
from agent_architecture.memory.items import MessageItem, ToolCallItem, ToolResultItem


async def test_add_message():
    context = ChatContext()
    await context.add_message("user", "Hello")
    assert len(context.items) == 1
    assert isinstance(context.items[0], MessageItem)


async def test_add_tool_call_and_result():
    context = ChatContext()
    tool_call = await context.add_tool_call("my_tool", {"key": "value"})
    await context.add_tool_result(tool_call.id, {"result": "ok"})
    assert len(context.items) == 2
    assert isinstance(context.items[0], ToolCallItem)
    assert isinstance(context.items[1], ToolResultItem)


async def test_messages_for_llm_includes_tool_calls():
    context = ChatContext()
    await context.add_message("user", "Hello")
    tool_call = await context.add_tool_call("my_tool", {"key": "value"})
    await context.add_tool_result(tool_call.id, {"result": "ok"})

    messages = context.messages_for_llm()
    roles = [m["role"] for m in messages]

    assert "user" in roles
    assert "assistant" in roles
    assert "tool" in roles


async def test_messages_for_llm_excludes_events():
    context = ChatContext()
    await context.add_message("user", "Hello")
    await context.add_event("some_event", {"data": "value"})

    messages = context.messages_for_llm()
    assert len(messages) == 1


async def test_compact_keeps_system_messages():
    context = ChatContext(max_context_items=2)
    await context.add_message("system", "You are helpful")
    await context.add_message("user", "message 1")
    await context.add_message("user", "message 2")
    await context.add_message("user", "message 3")

    system_items = [i for i in context.items if isinstance(i, MessageItem) and i.role == "system"]
    assert len(system_items) == 1


async def test_compact_respects_max_items():
    context = ChatContext(max_context_items=3)
    for i in range(10):
        await context.add_message("user", f"message {i}")

    non_system = [i for i in context.items if not (isinstance(i, MessageItem) and i.role == "system")]
    assert len(non_system) <= 3


async def test_auto_compact_triggered_on_append():
    context = ChatContext(max_context_items=3)
    for i in range(10):
        await context.add_message("user", f"message {i}")

    assert len(context.items) <= 4


def test_session_store_save_and_load(tmp_path):
    store = SessionStore(db_path=tmp_path / "test.db")
    items = [{"kind": "message", "role": "user", "content": "Hello"}]
    store.save("session-1", items)

    loaded = store.load("session-1")
    assert loaded == items


def test_session_store_load_missing_returns_none(tmp_path):
    store = SessionStore(db_path=tmp_path / "test.db")
    assert store.load("nonexistent") is None


def test_session_store_delete(tmp_path):
    store = SessionStore(db_path=tmp_path / "test.db")
    store.save("session-1", [{"kind": "message"}])
    store.delete("session-1")
    assert store.load("session-1") is None


async def test_resume_restores_items(tmp_path):
    store = SessionStore(db_path=tmp_path / "test.db")

    context = ChatContext(session_id="session-1")
    await context.add_message("user", "Hello")
    await context.add_message("assistant", "Hi there")
    store.save("session-1", context.dump())

    new_context = ChatContext(session_id="session-1")
    result = new_context.resume(store)

    assert result is True
    assert len(new_context.items) == 2


def test_resume_returns_false_for_missing_session(tmp_path):
    store = SessionStore(db_path=tmp_path / "test.db")
    context = ChatContext(session_id="nonexistent")
    assert context.resume(store) is False
