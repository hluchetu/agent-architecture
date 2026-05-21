from __future__ import annotations

import logging

import structlog

from agent_architecture.observability.events import ObservabilityEvent


def configure_logging(level: int = logging.INFO) -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    logging.basicConfig(level=level)


class EventLog:
    """Records observability events as structured JSON logs."""

    def __init__(self) -> None:
        self._log = structlog.get_logger("agent_architecture")

    def emit(self, event: ObservabilityEvent) -> None:
        log = self._log.bind(
            event=event.name,
            session_id=event.session_id,
            turn_id=event.turn_id,
        )

        if event.name == "memory.item_added":
            log.info(
                event.name,
                item_kind=event.payload.get("item_kind"),
                item_id=event.payload.get("item_id"),
            )
            return

        if event.name == "memory.context_built":
            log.info(
                event.name,
                message_count=event.payload.get("message_count"),
                total_item_count=event.payload.get("total_item_count"),
            )
            return

        if event.name == "memory.compacted":
            log.info(
                event.name,
                before_count=event.payload.get("before_count"),
                after_count=event.payload.get("after_count"),
                max_items=event.payload.get("max_items"),
            )
            return

        if event.name == "llm.request":
            log.info(
                event.name,
                model=event.payload.get("model"),
                message_count=event.payload.get("message_count"),
            )
            return

        if event.name == "llm.response":
            log.info(
                event.name,
                model=event.payload.get("model"),
                content=event.payload.get("content"),
            )
            return

        log.info(event.name, **event.payload)
