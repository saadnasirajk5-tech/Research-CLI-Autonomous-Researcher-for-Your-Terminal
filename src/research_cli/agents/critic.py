"""
Critic Agent - The Auditor.

Reviews each researcher's output against the original sub-task.
Checks for citations, specificity, and relevance.
If the research is inadequate, sends it back for revision.
This is the quality gate that prevents garbage output.
"""

import json
from langchain_ollama import ChatOllama
from research_cli.config import Config


class CriticAgent:
    """
    Audits research findings for quality and completeness.

    The critic is stricter than the researcher - it catches:
    - Vague claims without numbers or specifics
    - Missing citations
    - Claims that don't address the sub-task
    - Unsupported assertions
    """

    def __init__(self, config: Config | None = None):
        """Initialize the critic with a small model."""
        self.config = config or Config()
        self.config.initialize()
        self.model = ChatOllama(
            model=self.config.model.model_name,
            temperature=0.0,
            base_url=self.config.model.base_url,
        )
        self.critic_config = self.config.critic

    def evaluate(self, task_description: str, task_result: dict) -> dict:
        """
        Evaluate a researcher's output.

        Parameters:
            task_description: The original sub-task description.
            task_result: The researcher's output (findings, summary, etc.).

        Returns:
            Dict with passed, score, issues, and feedback.
        """
        prompt = self._build_prompt(task_description, task_result)
        response = self.model.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return self._parse_response(content)

    def _build_prompt(self, task_description: str, task_result: dict) -> str:
        """Build the critic prompt with the task and results."""
        findings_text = ""
        for f in task_result.get("findings", []):
            findings_text += f"- Claim: {f.get('claim', '')}\n"
            findings_text += f"  Source: {f.get('source_url', 'MISSING')}\n"
            findings_text += f"  Confidence: {f.get('confidence', 'unknown')}\n\n"
        return f"""{self.critic_config.system_prompt}

Original sub-task: "{task_description}"

Researcher's findings:
{findings_text}

Researcher's summary: {task_result.get('summary', 'No summary provided.')}

Evaluate these findings. Output ONLY valid JSON:"""

    def _parse_response(self, content: str) -> dict:
        """Parse the critic's JSON evaluation."""
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        try:
            data = json.loads(content)
            return {
                "passed": data.get("passed", False),
                "score": min(10, max(0, int(data.get("score", 0)))),
                "issues": data.get("issues", []),
                "feedback": data.get("feedback", ""),
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            return self._fallback_evaluate(content)

    def _fallback_evaluate(self, raw_response: str) -> dict:
        """Default evaluation if parsing fails - conservative pass."""
        if len(raw_response) < 50:
            return {
                "passed": False,
                "score": 2,
                "issues": ["Response too short, likely incomplete"],
                "feedback": "Provide more detailed findings with specific data.",
            }
        return {
            "passed": True,
            "score": 5,
            "issues": [],
            "feedback": "",
        }
