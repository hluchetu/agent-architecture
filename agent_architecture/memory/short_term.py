from __future__ import annotations

from typing import Any
from uuid import uuid4

from agent_architecture.memory.items import (
    ContextItem,
    EventItem,
    MessageItem,
    Role,
    SummaryItem,
    ToolCallItem,
    ToolResultItem,
)
from agent_architecture.observability import EventLog, ObservabilityEvent


class ChatContext:
    """Short-term memory for one active agent session.

    This stores rich internal context items, but exposes a model-facing view
    through messages_for_llm(). That separation is important in production:
    not everything the system remembers should be sent directly to the model.
    """

    def __init__(
        self,
        session_id: str | None = None,
        *,
        event_log: EventLog | None = None,
    ) -> None:
        self.session_id = session_id or str(uuid4())
        self.items: list[ContextItem] = []
        self.event_log = event_log

    def _emit(
        self,
        name: str,
        context_item: ContextItem | None = None,
        **payload: Any,
    ) -> None:
        if self.event_log is None:
            return

        self.event_log.emit(
            ObservabilityEvent(
                name=name,
                session_id=self.session_id,
                turn_id=(
                    context_item.turn_id
                    if context_item
                    else payload.pop("turn_id", None)
                ),
                payload=payload,
            )
        )

    def _append(self, item: ContextItem) -> None:
        self.items.append(item)
        self._emit(
            "memory.item_added",
            item,
            item_kind=item.kind,
            item_id=item.id,
            item=item.model_dump(mode="json"),
        )

    def new_turn_id(self) -> str:
        return str(uuid4())

    def add_message(
        self,
        role: Role,
        content: str,
        *,
        turn_id: str | None = None,
    ) -> MessageItem:
        item = MessageItem(
            session_id=self.session_id,
            turn_id=turn_id,
            role=role,
            content=content,
        )
        self._append(item)
        return item

    def add_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        turn_id: str | None = None,
    ) -> ToolCallItem:
        item = ToolCallItem(
            session_id=self.session_id,
            turn_id=turn_id,
            name=name,
            arguments=arguments,
        )
        self._append(item)
        return item

    def add_tool_result(
        self,
        tool_call_id: str,
        result: dict[str, Any],
        *,
        turn_id: str | None = None,
        error: str | None = None,
    ) -> ToolResultItem:
        item = ToolResultItem(
            session_id=self.session_id,
            turn_id=turn_id,
            tool_call_id=tool_call_id,
            result=result,
            error=error,
        )
        self._append(item)
        return item

    def add_event(
        self,
        name: str,
        data: dict[str, Any] | None = None,
        *,
        turn_id: str | None = None,
    ) -> EventItem:
        item = EventItem(
            session_id=self.session_id,
            turn_id=turn_id,
            name=name,
            data=data or {},
        )
        self._append(item)
        return item

    def add_summary(self, content: str) -> SummaryItem:
        item = SummaryItem(session_id=self.session_id, content=content)
        self._append(item)
        return item

    def messages_for_llm(self) -> list[dict[str, str]]:
        """Build the chat-completion view sent to the model."""

        messages: list[dict[str, str]] = []

        for item in self.items:
            if isinstance(item, SummaryItem):
                messages.append(
                    {
                        "role": "system",
                        "content": f"Conversation summary: {item.content}",
                    }
                )
            elif isinstance(item, MessageItem):
                messages.append({"role": item.role, "content": item.content})

        self._emit(
            "memory.context_built",
            message_count=len(messages),
            total_item_count=len(self.items),
            messages=messages,
        )
        return messages

    def compact(self, max_items: int) -> None:
        """Keep system messages and the latest context items.

        This is a simple sliding-window compaction strategy. Later, this can be
        replaced with summarization or importance-based memory.
        """

        system_messages = [
            item
            for item in self.items
            if isinstance(item, MessageItem) and item.role == "system"
        ]
        recent_items = self.items[-max_items:]

        before_count = len(self.items)
        self.items = system_messages + [
            item for item in recent_items if item not in system_messages
        ]
        self._emit(
            "memory.compacted",
            before_count=before_count,
            after_count=len(self.items),
            max_items=max_items,
        )

    def dump(self) -> list[dict[str, Any]]:
        """Return a serializable view useful for debugging and tests."""

        return [item.model_dump(mode="json") for item in self.items]
