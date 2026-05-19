"""
Planner Agent - The Strategist.

Takes a broad user query and breaks it into a DAG of independent sub-tasks.
Each sub-task has a specific search query and can run in parallel (no dependencies).
Uses a small model efficiently by giving it one focused job.
"""

import json
from langchain_ollama import ChatOllama
from research_cli.config import Config, PlannerConfig


class PlannerAgent:
    """
    Breaks a research query into a directed acyclic graph of sub-tasks.

    The planner thinks about what angles to cover:
    - Historical context
    - Current state
    - Comparisons
    - Opposing viewpoints
    - Recent developments
    """

    def __init__(self, config: Config | None = None):
        """Initialize the planner with a small model."""
        self.config = config or Config()
        self.config.initialize()
        self.model = ChatOllama(
            model=self.config.model.model_name,
            temperature=self.config.model.temperature,
            base_url=self.config.model.base_url,
        )
        self.planner_config = self.config.planner

    def plan(self, query: str) -> list[dict]:
        """
        Generate sub-tasks for a research query.

        Parameters:
            query: The user's broad research question.

        Returns:
            List of sub-task dicts with id, description, search_query, dependencies.
        """
        prompt = self._build_prompt(query)
        response = self.model.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        subtasks = self._parse_response(content)
        return subtasks

    def _build_prompt(self, query: str) -> str:
        """Build the planner prompt with the user's query."""
        return f"""{self.planner_config.system_prompt}

User's research query: "{query}"

Break this into {self.planner_config.min_subtasks}-{self.planner_config.max_subtasks} specific sub-tasks.
Output ONLY valid JSON:"""

    def _parse_response(self, content: str) -> list[dict]:
        """
        Parse the model's JSON response into sub-tasks.

        Handles common issues like markdown code blocks wrapping the JSON.
        Falls back to a default plan if parsing fails.
        """
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        try:
            data = json.loads(content)
            subtasks = data.get("subtasks", data)
            if isinstance(subtasks, list) and len(subtasks) > 0:
                for task in subtasks:
                    if "dependencies" not in task:
                        task["dependencies"] = []
                return subtasks
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return self._fallback_plan(content)

    def _fallback_plan(self, raw_response: str) -> list[dict]:
        """
        Create a default plan if the model's output can't be parsed.

        Ensures the research pipeline always has tasks to execute.
        """
        return [
            {
                "id": "t1",
                "description": f"Research the core topic: {raw_response[:100]}",
                "search_query": raw_response[:100],
                "dependencies": [],
            },
            {
                "id": "t2",
                "description": "Find recent developments and statistics",
                "search_query": f"{raw_response[:80]} latest news statistics 2024 2025",
                "dependencies": [],
            },
            {
                "id": "t3",
                "description": "Compare different viewpoints and perspectives",
                "search_query": f"{raw_response[:80]} debate pros cons perspectives",
                "dependencies": [],
            },
        ]
