"""
Scraper Tool - Web page content extractor.

Primary: Crawl4AI for JS-rendered pages.
Fallback: urllib + html2text for static pages.
No API keys needed. Runs entirely locally.
"""

import asyncio
import urllib.request
import urllib.error
import re
from research_cli.config import Config

CRAWL4AI_AVAILABLE = False
try:
    from crawl4ai import AsyncWebCrawler
    CRAWL4AI_AVAILABLE = True
except ImportError:
    pass


def scrape_page(url: str, max_tokens: int = 4000, config: Config | None = None) -> str:
    """
    Scrape a single URL and return clean text content.

    Parameters:
        url: The URL to scrape.
        max_tokens: Maximum tokens to return (truncates long pages).
        config: Optional config (unused but kept for API consistency).

    Returns:
        Clean text from the page, or an error message.
    """
    if not url:
        return "Error: Empty URL provided."

    try:
        if CRAWL4AI_AVAILABLE:
            result = asyncio.run(_async_scrape_crawl4ai(url))
            if result:
                text = result
                if len(text) > max_tokens * 4:
                    text = text[: max_tokens * 4] + "\n... [truncated]"
                return text
        return _fallback_scrape(url, max_tokens)
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"


async def _async_scrape_crawl4ai(url: str) -> str | None:
    """Scrape using Crawl4AI for JS-rendered pages."""
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url)
            if result and result.markdown:
                return result.markdown
            return None
    except Exception:
        return None


def _fallback_scrape(url: str, max_tokens: int) -> str:
    """
    Fallback scraper using urllib and regex.

    Fetches the raw HTML and strips tags to get readable text.
    Works for static pages without JavaScript rendering.
    """
    import gzip
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (research-cli/0.1.0)"},
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            raw = response.read()
            encoding = response.headers.get("Content-Encoding", "")
            if "gzip" in encoding.lower():
                html = gzip.decompress(raw).decode("utf-8", errors="replace")
            else:
                html = raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"Error: HTTP {e.code} - {e.reason}"
    except urllib.error.URLError as e:
        return f"Error: Could not reach {url} - {e.reason}"
    except Exception as e:
        return f"Error: {str(e)}"

    text = _html_to_text(html)
    if len(text) > max_tokens * 4:
        text = text[: max_tokens * 4] + "\n... [truncated]"
    return text


def _html_to_text(html: str) -> str:
    """
    Convert HTML to plain text.

    Removes script/style blocks, strips tags, collapses whitespace.
    """
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#\d+;", "", text)
    text = re.sub(r"\s+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    return text.strip()
