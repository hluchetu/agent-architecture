from __future__ import annotations

from agent_architecture.agent import Agent
from agent_architecture.config import Settings
from agent_architecture.llm import LLMClient, create_llm_client
from agent_architecture.observability import EventLog
from agent_architecture.observability.tokens import TokenTracker
from agent_architecture.orchestrator import Orchestrator
from agent_architecture.prompts import REACT_SYSTEM_PROMPT
from agent_architecture.tools import MemoryLookupTool


ROUTER_PROMPT = """Classify the following task into one of three categories:

- simple: a direct question that needs no tools or planning
- tool: a question that needs a tool lookup but no planning
- complex: a multi-step task that needs planning and multiple steps

Return only one word: simple, tool, or complex."""


class Router:
    """Routes tasks to the right handler based on complexity."""

    def __init__(
        self,
        settings: Settings | None = None,
        event_log: EventLog | None = None,
        token_tracker: TokenTracker | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.event_log = event_log
        self.token_tracker = token_tracker
        self.llm: LLMClient = create_llm_client(
            self.settings,
            event_log=self.event_log,
            token_tracker=self.token_tracker,
        )

    async def _classify(self, task: str) -> str:
        response = await self.llm.chat([
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": task},
        ])
        return response.content.strip().lower()

    async def run(self, task: str) -> str:
        classification = await self._classify(task)
        print(f"[router] classified as: {classification}")

        if classification == "complex":
            orchestrator = Orchestrator(
                settings=self.settings,
                event_log=self.event_log,
                token_tracker=self.token_tracker,
            )
            return await orchestrator.run(task)

        if classification == "tool":
            agent = Agent(
                name="tool-agent",
                system_prompt="",
                settings=self.settings,
                event_log=self.event_log,
                token_tracker=self.token_tracker,
            )
            await agent.initialize(REACT_SYSTEM_PROMPT)
            return await agent.run(task)

        response = await self.llm.chat([
            {"role": "user", "content": task},
        ])
        return response.content
