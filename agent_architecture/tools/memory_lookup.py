from __future__ import annotations

from typing import Any

from agent_architecture.tools.tool import Tool


class MemoryLookupTool(Tool):
    name = "lookup_memory_definition"
    description = "Looks up a definition for a given memory-related topic."
    parameters = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The memory topic to look up.",
            }
        },
        "required": ["topic"],
    }

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        topic = arguments.get("topic", "").lower()
        definitions = {
            "short-term memory": "Temporary context used during the active session.",
            "long-term memory": "Persistent knowledge stored across sessions.",
            "episodic memory": "Records of past conversations and events.",
            "semantic memory": "Facts and knowledge retrieved by similarity.",
            "working memory": "Active task state tracked mid-session.",
        }
        definition = definitions.get(topic, f"No definition found for '{topic}'.")
        return {"definition": definition}
