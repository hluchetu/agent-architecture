from __future__ import annotations

from agent_architecture.observability.events import ObservabilityEvent


class EventLog:
    """Records observability events in a human-readable format."""

    def emit(self, event: ObservabilityEvent) -> None:
        if event.name == "memory.item_added":
            self._print_memory_item(event)
            return

        if event.name == "memory.context_built":
            print(
                "[memory.context_built] "
                f"messages={event.payload['message_count']} "
                f"items={event.payload['total_item_count']}"
            )
            return

        if event.name == "memory.compacted":
            print(
                "[memory.compacted] "
                f"before={event.payload['before_count']} "
                f"after={event.payload['after_count']} "
                f"max_items={event.payload['max_items']}"
            )
            return

        if event.name == "llm.request":
            print(
                "[llm.request] "
                f"model={event.payload['model']} "
                f"messages={event.payload['message_count']}"
            )
            return

        if event.name == "llm.response":
            print(
                "[llm.response] "
                f"model={event.payload['model']} "
                f"content={event.payload['content']!r}"
            )
            return

        print(f"[{event.name}] {event.payload}")

    def _print_memory_item(self, event: ObservabilityEvent) -> None:
        item = event.payload["item"]
        kind = event.payload["item_kind"]
        turn = event.turn_id or "-"

        if kind == "message":
            print(
                "[memory.item_added] "
                f"kind=message turn={turn} "
                f"role={item['role']} "
                f"content={item['content']!r}"
            )
            return

        if kind == "tool_call":
            print(
                "[memory.item_added] "
                f"kind=tool_call turn={turn} "
                f"name={item['name']} "
                f"arguments={item['arguments']}"
            )
            return

        if kind == "tool_result":
            print(
                "[memory.item_added] "
                f"kind=tool_result turn={turn} "
                f"tool_call_id={item['tool_call_id']} "
                f"result={item['result']} "
                f"error={item['error']}"
            )
            return

        if kind == "event":
            print(
                "[memory.item_added] "
                f"kind=event turn={turn} "
                f"name={item['name']} "
                f"data={item['data']}"
            )
            return

        print(f"[memory.item_added] kind={kind} turn={turn} item={item}")
