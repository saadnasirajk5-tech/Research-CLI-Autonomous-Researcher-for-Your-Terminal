"""
Search Tool - DuckDuckGo wrapper.

Provides web search without API keys.
Returns titles, snippets, and URLs for further scraping.
"""

DDGS_AVAILABLE = False
DDGS_CLASS = None

try:
    from ddgs import DDGS
    DDGS_CLASS = DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_CLASS = DDGS
        DDGS_AVAILABLE = True
    except ImportError:
        pass


def search_web(query: str, max_results: int = 5, config=None) -> list[dict]:
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
    if not DDGS_AVAILABLE:
        return [{"title": "Search Unavailable", "snippet": "Install ddgs: pip install ddgs", "url": ""}]
    try:
        with DDGS_CLASS() as ddgs:
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
