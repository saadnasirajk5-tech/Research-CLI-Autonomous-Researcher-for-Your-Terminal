"""
Search Tool - DuckDuckGo wrapper.

Provides web search without API keys.
Returns titles, snippets, and URLs for further scraping.
"""

from duckduckgo_search import DDGS
from research_cli.config import Config


def search_web(query: str, max_results: int = 5, config: Config | None = None) -> list[dict]:
    """
    Search the web using DuckDuckGo.

    Parameters:
        query: The search query string.
        max_results: Maximum number of results to return.
        config: Optional config (unused but kept for API consistency).

    Returns:
        List of dicts with keys: title, snippet, url.
    """
    results = []
    try:
        with DDGS() as ddgs:
            ddg_results = list(ddgs.text(query, max_results=max_results))
            for r in ddg_results:
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                })
    except Exception as e:
        results.append({
            "title": "Search Error",
            "snippet": f"Could not perform search: {str(e)}",
            "url": "",
        })
    return results
