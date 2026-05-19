"""
Researcher Agent - The Specialist.

Runs in parallel for each sub-task. Given one specific task and search results,
extracts facts, statistics, and quotes with citations.
Small models excel at this narrow, focused task.
"""

import json
from langchain_ollama import ChatOllama
from research_cli.config import Config
from research_cli.tools.search import search_web
from research_cli.tools.scraper import scrape_page


class ResearcherAgent:
    """
    Executes a single research sub-task.

    Workflow:
    1. Search the web for the task's query.
    2. Scrape the top pages for detailed content.
    3. Use the LLM to extract structured findings.
    4. Return findings with citations.
    """

    def __init__(self, config: Config | None = None):
        """Initialize the researcher with a small model."""
        self.config = config or Config()
        self.config.initialize()
        self.model = ChatOllama(
            model=self.config.model.model_name,
            temperature=self.config.model.temperature,
            base_url=self.config.model.base_url,
        )
        self.researcher_config = self.config.researcher

    def research(self, task_id: str, description: str, search_query: str) -> dict:
        """
        Execute a research sub-task.

        Parameters:
            task_id: The unique ID of this sub-task.
            description: Human-readable description of the task.
            search_query: The search query to use.

        Returns:
            Dict with findings, summary, and needs_more_research flag.
        """
        search_results = search_web(
            search_query,
            max_results=self.researcher_config.max_results_per_search,
            config=self.config,
        )
        scraped_content = []
        for result in search_results[: self.researcher_config.max_pages_to_scrape]:
            if result.get("url"):
                content = scrape_page(
                    result["url"],
                    max_tokens=self.researcher_config.max_tokens_per_page,
                    config=self.config,
                )
                scraped_content.append({
                    "url": result["url"],
                    "title": result.get("title", ""),
                    "content": content,
                })
        prompt = self._build_prompt(description, search_query, scraped_content)
        response = self.model.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return self._parse_response(content, task_id, scraped_content)

    def _build_prompt(self, description: str, search_query: str, sources: list[dict]) -> str:
        """Build the researcher prompt with search results."""
        sources_text = ""
        for i, src in enumerate(sources):
            sources_text += f"\n--- Source {i + 1}: {src['title']} ---\n"
            sources_text += f"URL: {src['url']}\n"
            sources_text += f"{src['content'][:2000]}\n"
        return f"""{self.researcher_config.system_prompt}

Sub-task: {description}
Search query used: {search_query}

Available sources:
{sources_text}

Extract findings from the sources above. Output ONLY valid JSON:"""

    def _parse_response(self, content: str, task_id: str, sources: list[dict]) -> dict:
        """Parse the model's response into structured findings."""
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        try:
            data = json.loads(content)
            findings = []
            for f in data.get("findings", []):
                findings.append({
                    "task_id": task_id,
                    "claim": f.get("claim", ""),
                    "source_url": f.get("source_url", self._find_matching_url(f.get("claim", ""), sources)),
                    "confidence": f.get("confidence", "medium"),
                    "validated": False,
                })
            return {
                "task_id": task_id,
                "findings": findings,
                "summary": data.get("summary", ""),
                "needs_more_research": data.get("needs_more_research", False),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            return self._fallback_result(content, task_id, sources)

    def _find_matching_url(self, claim: str, sources: list[dict]) -> str:
        """Find the most likely source URL for a claim."""
        for src in sources:
            if src.get("title", "").lower() in claim.lower():
                return src["url"]
        return sources[0]["url"] if sources else ""

    def _fallback_result(self, raw_response: str, task_id: str, sources: list[dict]) -> dict:
        """Create a fallback result if parsing fails."""
        url = sources[0]["url"] if sources else ""
        return {
            "task_id": task_id,
            "findings": [{
                "task_id": task_id,
                "claim": raw_response[:300],
                "source_url": url,
                "confidence": "low",
                "validated": False,
            }],
            "summary": raw_response[:200],
            "needs_more_research": True,
        }
