"""
DuckDuckGo web search wrapper.

Uses the new `ddgs` package (renamed from `duckduckgo_search`).
Used by the agent as an external source when the local knowledge base doesn't
have enough information to answer a query.

Returns a list of result dicts with: title, url, snippet, full_text (best-effort).
"""
from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import List

import requests

logger = logging.getLogger(__name__)

# Prefer the new `ddgs` package; fall back to the legacy `duckduckgo_search` if
# that's what the user has installed. Both expose a `DDGS` class with the same API.
try:
    from ddgs import DDGS  # type: ignore
except ImportError:  # pragma: no cover
    from duckduckgo_search import DDGS  # type: ignore


def web_search(query: str, max_results: int = 5) -> List[dict]:
    """Run a DuckDuckGo text search and return structured results."""
    results: List[dict] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href") or r.get("link") or r.get("url", ""),
                        "snippet": r.get("body") or r.get("snippet", ""),
                    }
                )
    except Exception as e:
        logger.error("DuckDuckGo search failed: %s", e)
        return [{"title": "Search error", "url": "", "snippet": f"Web search failed: {e}"}]

    return results


def fetch_url_text(url: str, timeout: int = 8) -> str:
    """Best-effort fetch of a URL's text content. Strips HTML tags crudely."""
    if not url:
        return ""
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (multimodal-rag-bot)"},
        )
        resp.raise_for_status()
        html = resp.text

        class _Stripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts: List[str] = []

            def handle_data(self, data):
                t = data.strip()
                if t:
                    self.parts.append(t)

        s = _Stripper()
        s.feed(html)
        text = " ".join(s.parts)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:4000]
    except Exception as e:
        logger.warning("fetch_url_text failed for %s: %s", url, e)
        return ""


def search_and_fetch(query: str, max_results: int = 5) -> List[dict]:
    """Search + enrich each result with full-text (best-effort)."""
    base = web_search(query, max_results=max_results)
    for r in base:
        r["full_text"] = fetch_url_text(r.get("url", ""))
    return base
