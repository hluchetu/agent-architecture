from __future__ import annotations

from agent_architecture.agent import Agent
from agent_architecture.config import Settings
from agent_architecture.observability import EventLog
from agent_architecture.prompts import REACT_SYSTEM_PROMPT
from agent_architecture.tools import MemoryLookupTool


class Orchestrator:
    """Coordinates multiple worker agents to complete a complex task."""

    def __init__(
        self,
        settings: Settings | None = None,
        event_log: EventLog | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.event_log = event_log

    async def run(self, task: str) -> str:
        # Step 1 — plan the task into subtasks
        planner = Agent(
            name="planner",
            system_prompt="Break the given task into clear independent subtasks. Return a numbered list only.",
            settings=self.settings,
            event_log=self.event_log,
        )
        plan_text = await planner.run(task)
        steps = [
            line.strip()
            for line in plan_text.splitlines()
            if line.strip() and line.strip()[0].isdigit()
        ]
        print(f"[orchestrator] plan has {len(steps)} steps")

        # Step 2 — run a worker agent for each step
        results = []
        for step in steps:
            print(f"[orchestrator] running step: {step}")
            worker = Agent(
                name="worker",
                system_prompt=REACT_SYSTEM_PROMPT,
                settings=self.settings,
                tools=[MemoryLookupTool()],
                event_log=self.event_log,
            )
            result = await worker.run(step)
            results.append(f"{step}\n{result}")

        # Step 3 — summarize all results into one final answer
        summarizer = Agent(
            name="summarizer",
            system_prompt="You combine results from multiple agents into one clear final answer.",
            settings=self.settings,
            event_log=self.event_log,
        )
        combined = "\n\n".join(results)
        return await summarizer.run(f"Combine these results:\n\n{combined}")
