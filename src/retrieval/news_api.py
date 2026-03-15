#!/usr/bin/env python3
"""
news_api.py
-----------
Fetch relevant news articles from NewsAPI.org.

Requires a free API key from https://newsapi.org/
Set it in environment variable:  NEWSAPI_KEY=your_key_here
Or pass it directly to search_news().
"""
from __future__ import annotations
import os
import time
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

NEWSAPI_BASE = "https://newsapi.org/v2/everything"
DEFAULT_TIMEOUT = 10  # seconds

__all__ = ["search_news"]


def search_news(
    query: str,
    api_key: Optional[str] = None,
    max_results: int = 5,
    language: str = "en",
    sort_by: str = "relevancy",
) -> List[Dict[str, Any]]:
    """
    Search NewsAPI for articles matching *query*.

    Parameters
    ----------
    query : str
        Search query (extracted claim or headline).
    api_key : str, optional
        NewsAPI key. Falls back to ``NEWSAPI_KEY`` env var.
    max_results : int
        How many articles to return (max 100 per request).
    language : str
        Language filter (default "en").
    sort_by : str
        "relevancy" | "popularity" | "publishedAt"

    Returns
    -------
    list[dict]
        Each dict has keys: title, description, url, source, publishedAt.
        Returns empty list on error or missing key.
    """
    key = api_key or os.environ.get("NEWSAPI_KEY", "")
    if not key:
        logger.warning(
            "No NewsAPI key found. Set NEWSAPI_KEY env variable or pass api_key=. "
            "Returning empty results."
        )
        return []

    params = {
        "q": query,
        "language": language,
        "sortBy": sort_by,
        "pageSize": min(max_results, 100),
        "apiKey": key,
    }

    try:
        resp = requests.get(NEWSAPI_BASE, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            logger.error("NewsAPI error: %s", data.get("message", "unknown"))
            return []

        articles = data.get("articles", [])
        # Normalize to consistent schema
        results = []
        for a in articles[:max_results]:
            results.append(
                {
                    "title": a.get("title") or "",
                    "description": a.get("description") or "",
                    "url": a.get("url") or "",
                    "source": (a.get("source") or {}).get("name", "Unknown"),
                    "publishedAt": a.get("publishedAt") or "",
                    "content": a.get("content") or "",
                }
            )
        return results

    except requests.RequestException as exc:
        logger.error("NewsAPI request failed: %s", exc)
        return []
