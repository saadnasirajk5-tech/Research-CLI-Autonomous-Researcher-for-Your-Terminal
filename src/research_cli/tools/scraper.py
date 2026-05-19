"""
Scraper Tool - Crawl4AI wrapper.

Scrapes web pages and converts them to clean Markdown.
No API keys needed. Runs entirely locally.
"""

import asyncio
from research_cli.config import Config


def scrape_page(url: str, max_tokens: int = 4000, config: Config | None = None) -> str:
    """
    Scrape a single URL and return clean Markdown content.

    Parameters:
        url: The URL to scrape.
        max_tokens: Maximum tokens to return (truncates long pages).
        config: Optional config (unused but kept for API consistency).

    Returns:
        Clean Markdown text from the page, or an error message.
    """
    if not url:
        return "Error: Empty URL provided."

    try:
        result = asyncio.run(_async_scrape(url))
        if result and result.get("markdown"):
            text = result["markdown"]
            if len(text) > max_tokens * 4:
                text = text[: max_tokens * 4] + "\n... [truncated]"
            return text
        return "Error: Could not extract content from page."
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"


async def _async_scrape(url: str) -> dict | None:
    """
    Async wrapper for Crawl4AI.

    Uses Crawl4AI to fetch and parse the page.
    Returns a dict with the markdown content.
    """
    try:
        from crawl4ai import AsyncWebCrawler
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url)
            if result and result.markdown:
                return {"markdown": result.markdown, "url": url}
            return None
    except ImportError:
        return {"markdown": f"Crawl4AI not installed. Raw URL: {url}", "url": url}
    except Exception:
        return None
