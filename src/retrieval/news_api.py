#!/usr/bin/env python3
"""
news_api.py
-----------
Fetch news articles from either:
  • NewsAPI.org       (key looks like: abc123def456...  32 hex chars)
  • NewsData.io       (key looks like: pub_xxxxx...)

Auto-detects which provider to use from the key format.
"""
from __future__ import annotations
import os
import logging
import re
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 12

__all__ = ["search_news", "detect_provider"]

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------
def detect_provider(key: str) -> str:
    """
    Return 'newsdata' if the key starts with 'pub_', else 'newsapi'.
    """
    if key.strip().startswith("pub_"):
        return "newsdata"
    return "newsapi"


def _clean_key(key: str) -> str:
    """Strip accidental whitespace, URL prefixes, quotes."""
    key = key.strip().strip('"').strip("'")
    # If the user pasted a full URL, extract just the key param
    m = re.search(r"apikey=([^\s&]+)", key, re.IGNORECASE)
    if m:
        key = m.group(1)
    return key


# ---------------------------------------------------------------------------
# NewsAPI.org  (free tier: dev only, no production)
# ---------------------------------------------------------------------------
NEWSAPI_BASE = "https://newsapi.org/v2/everything"

def _search_newsapi(query: str, key: str, max_results: int, language: str) -> List[Dict]:
    params = {
        "q": query[:500],          # NewsAPI max q length
        "language": language,
        "sortBy": "relevancy",
        "pageSize": min(max_results, 10),   # free tier caps at 10 per request
        "apiKey": key,
    }
    try:
        resp = requests.get(NEWSAPI_BASE, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            logger.error("NewsAPI error: %s", data.get("message"))
            return []
        return _normalize_newsapi(data.get("articles", [])[:max_results])
    except requests.RequestException as exc:
        logger.error("NewsAPI request failed: %s", exc)
        return []


def _normalize_newsapi(articles: list) -> List[Dict]:
    out = []
    for a in articles:
        out.append({
            "title":       a.get("title") or "",
            "description": a.get("description") or "",
            "url":         a.get("url") or "",
            "source":      (a.get("source") or {}).get("name", "Unknown"),
            "publishedAt": a.get("publishedAt") or "",
            "content":     a.get("content") or "",
        })
    return out


# ---------------------------------------------------------------------------
# NewsData.io  (free tier: 200 credits/day, key starts with pub_)
# ---------------------------------------------------------------------------
NEWSDATA_BASE = "https://newsdata.io/api/1/latest"

def _search_newsdata(query: str, key: str, max_results: int, language: str) -> List[Dict]:
    params = {
        "apikey": key,
        "q":      query[:512],
        "language": language,
    }
    try:
        resp = requests.get(NEWSDATA_BASE, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            logger.error("NewsData error: %s", data.get("message"))
            return []
        results = data.get("results") or []
        return _normalize_newsdata(results[:max_results])
    except requests.RequestException as exc:
        logger.error("NewsData request failed: %s", exc)
        return []


def _normalize_newsdata(articles: list) -> List[Dict]:
    out = []
    for a in articles:
        out.append({
            "title":       a.get("title") or "",
            "description": a.get("description") or "",
            "url":         a.get("link") or "",
            "source":      a.get("source_id") or a.get("source_name") or "Unknown",
            "publishedAt": a.get("pubDate") or "",
            "content":     a.get("content") or "",
        })
    return out


# ---------------------------------------------------------------------------
# Public unified entry point
# ---------------------------------------------------------------------------
def search_news(
    query: str,
    api_key: Optional[str] = None,
    max_results: int = 5,
    language: str = "en",
    sort_by: str = "relevancy",   # kept for API compat, used by NewsAPI only
) -> List[Dict[str, Any]]:
    """
    Search for news articles matching *query*.
    Auto-detects NewsAPI vs NewsData from the key format.
    Returns [] on error or missing key — never raises.
    """
    raw_key = api_key or os.environ.get("NEWSAPI_KEY", "") or os.environ.get("NEWSDATA_KEY", "")
    if not raw_key or not raw_key.strip():
        logger.warning("No news API key provided. Returning empty results.")
        return []

    key = _clean_key(raw_key)
    provider = detect_provider(key)
    logger.info("Using news provider: %s", provider)

    if provider == "newsdata":
        return _search_newsdata(query, key, max_results, language)
    else:
        return _search_newsapi(query, key, max_results, language)