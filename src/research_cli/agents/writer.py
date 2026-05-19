"""
Writer Agent - The Synthesizer.

Compiles all verified findings into a comprehensive research report.
Only runs after the Critic confirms all tasks are complete.
Uses structured output for consistent report formatting.
"""

from langchain_ollama import ChatOllama
from research_cli.config import Config


class WriterAgent:
    """
    Synthesizes verified findings into a professional research report.

    The writer receives all validated findings organized by sub-task.
    It produces a structured report with:
    - Executive summary
    - Key findings (bullet points)
    - Detailed analysis (prose)
    - Source list with URLs
    """

    def __init__(self, config: Config | None = None):
        """Initialize the writer with a small model."""
        self.config = config or Config()
        self.config.initialize()
        self.model = ChatOllama(
            model=self.config.model.model_name,
            temperature=0.3,
            base_url=self.config.model.base_url,
        )
        self.writer_config = self.config.writer

    def write_report(self, query: str, task_results: list[dict]) -> str:
        """
        Generate the final research report.

        Parameters:
            query: The original user query.
            task_results: List of all completed task results with findings.

        Returns:
            The formatted research report as a string.
        """
        prompt = self._build_prompt(query, task_results)
        response = self.model.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def _build_prompt(self, query: str, task_results: list[dict]) -> str:
        """Build the writer prompt with all findings."""
        findings_text = ""
        for i, result in enumerate(task_results):
            findings_text += f"\n=== Task {i + 1}: {result.get('task_id', '')} ===\n"
            findings_text += f"Description: {result.get('description', '')}\n\n"
            for f in result.get("findings", []):
                validated = "[VERIFIED]" if f.get("validated") else "[UNVERIFIED]"
                findings_text += f"  {validated} {f.get('claim', '')}\n"
                findings_text += f"    Source: {f.get('source_url', 'N/A')}\n"
                findings_text += f"    Confidence: {f.get('confidence', 'unknown')}\n\n"
            findings_text += f"Summary: {result.get('summary', '')}\n"
        return f"""{self.writer_config.system_prompt}

Original research query: "{query}"

Verified findings organized by sub-task:
{findings_text}

Write a comprehensive research report. Include:
1. Executive Summary (2-3 paragraphs)
2. Key Findings (bullet points with citations)
3. Detailed Analysis (organized by theme)
4. Sources (list all URLs used)
5. Limitations and Uncertainties

Report:"""
