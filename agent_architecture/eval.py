from __future__ import annotations

from agent_architecture.config import Settings
from agent_architecture.llm import OllamaClient


JUDGE_PROMPT = """You are an expert evaluator scoring an AI agent's answer.

Question: {question}

Answer: {answer}

Rubric: {rubric}

Score the answer from 1 to 5:
1 - completely wrong or missing
2 - partially correct but major gaps
3 - mostly correct with minor gaps
4 - correct and complete
5 - excellent, thorough, and clear

Return only a single integer between 1 and 5. Nothing else."""


class Evaluator:
    """Scores agent answers against a rubric using an LLM as judge."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.llm = OllamaClient(self.settings)

    async def evaluate(
        self,
        question: str,
        answer: str,
        rubric: str,
    ) -> int:
        prompt = JUDGE_PROMPT.format(
            question=question,
            answer=answer,
            rubric=rubric,
        )
        response = await self.llm.chat([
            {"role": "user", "content": prompt},
        ])

        try:
            score = int(response.content.strip()[0])
            return max(1, min(5, score))
        except (ValueError, IndexError):
            return 1
