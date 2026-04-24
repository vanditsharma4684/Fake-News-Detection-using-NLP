#!/usr/bin/env python3
"""
news_api.py
-----------
Fetch news articles from either:
  • NewsAPI.org       (key: 32 hex chars)
  • NewsData.io       (key: starts with pub_)

Auto-detects provider from key format.

Fix applied:
  422 UNPROCESSABLE ENTITY was caused by sending full claim sentences (200+
  chars, curly quotes, em-dashes) directly as the API query parameter.
  NewsData.io rejects queries that:
    - Exceed ~100 characters
    - Contain non-ASCII characters (Unicode curly quotes, em-dashes, etc.)
    - Contain certain punctuation characters

  Solution: make_search_query() converts any raw claim text into a short
  (≤ 100 char), ASCII-only, stopword-filtered keyword phrase before it
  reaches any API call.
"""
from __future__ import annotations
import hashlib
import os
import logging
import re
import time
import threading
import unicodedata
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 12

__all__ = ["search_news", "detect_provider", "make_search_query"]

# ---------------------------------------------------------------------------
# Query sanitiser — fixes the 422 error
# ---------------------------------------------------------------------------

# Stopwords that waste query budget without adding search signal
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "that", "this", "these",
    "those", "it", "its", "said", "says", "saying", "told", "tell", "tells",
    "night", "morning", "day", "according", "after", "also", "about",
    "saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday",
}

# NewsData.io hard limits (from API docs + empirical testing)
_NEWSDATA_MAX_Q_CHARS = 100
_NEWSAPI_MAX_Q_CHARS  = 500


def make_search_query(
    text: str,
    max_words: int = 8,
    max_chars: int = _NEWSDATA_MAX_Q_CHARS,
) -> str:
    """
    Convert a raw claim sentence into a short, API-safe keyword query.

    Problems this solves:
      - NewsData 422: queries > ~100 chars are rejected
      - NewsData 422: non-ASCII chars (curly quotes \u2019, em-dash \u2014,
        accented letters) cause UNPROCESSABLE ENTITY even when URL-encoded
      - Poor results: sending a 200-char sentence returns irrelevant articles;
        5–8 keywords return far better results

    Steps:
      1. Unicode NFKD normalise → ASCII-only (curly ' → ', " → ", etc.)
      2. Strip all punctuation except spaces
      3. Remove common stopwords that waste query budget
      4. Take the first max_words content words
      5. Hard-cap at max_chars characters

    Parameters
    ----------
    text : str
        Raw claim sentence from claim_extractor.
    max_words : int
        Maximum number of words to include (default 8).
    max_chars : int
        Hard character cap (default 100 for NewsData compatibility).

    Returns
    -------
    str
        Clean, ASCII-only, short keyword query.
    """
    if not text:
        return ""

    # Step 1: Normalise unicode and drop non-ASCII
    normalised = unicodedata.normalize("NFKD", text)
    ascii_only  = normalised.encode("ascii", errors="ignore").decode("ascii")

    # Step 2: Remove punctuation (keep alphanumeric + spaces)
    no_punct = re.sub(r"[^\w\s]", " ", ascii_only)

    # Step 3: Collapse whitespace
    clean = re.sub(r"\s+", " ", no_punct).strip()

    # Step 4: Filter stopwords, keep meaningful content words
    words = [
        w for w in clean.split()
        if w.lower() not in _STOPWORDS and len(w) > 1
    ]

    # Step 5: Take first max_words, then cap chars
    query = " ".join(words[:max_words])
    return query[:max_chars].strip()


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def detect_provider(key: str) -> str:
    """Return 'newsdata' if key starts with 'pub_', else 'newsapi'."""
    return "newsdata" if key.strip().startswith("pub_") else "newsapi"


def _clean_key(key: str) -> str:
    """Strip accidental whitespace, URL prefixes, quotes."""
    key = key.strip().strip('"').strip("'")
    m = re.search(r"apikey=([^\s&]+)", key, re.IGNORECASE)
    if m:
        key = m.group(1)
    return key


# ---------------------------------------------------------------------------
# NewsAPI.org
# ---------------------------------------------------------------------------
NEWSAPI_BASE = "https://newsapi.org/v2/everything"


def _search_newsapi(query: str, key: str, max_results: int, language: str) -> List[Dict]:
    # Use spaCy-aware entity extraction when available, then ASCII-sanitise
    try:
        from src.preprocessing.claim_extractor import claim_to_query
        entity_query = claim_to_query(query, max_words=10)
        safe_query = make_search_query(entity_query or query, max_words=10, max_chars=_NEWSAPI_MAX_Q_CHARS)
    except Exception:
        safe_query = make_search_query(query, max_words=10, max_chars=_NEWSAPI_MAX_Q_CHARS)
    if not safe_query:
        logger.warning("NewsAPI: query sanitised to empty string — skipping.")
        return []

    params = {
        "q":        safe_query,
        "language": language,
        "sortBy":   "relevancy",
        "pageSize": min(max_results, 10),
        "apiKey":   key,
    }
    logger.debug("NewsAPI query: '%s'", safe_query)
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
    return [
        {
            "title":       a.get("title") or "",
            "description": a.get("description") or "",
            "url":         a.get("url") or "",
            "source":      (a.get("source") or {}).get("name", "Unknown"),
            "publishedAt": a.get("publishedAt") or "",
            "content":     a.get("content") or "",
        }
        for a in articles
    ]


# ---------------------------------------------------------------------------
# NewsData.io
# ---------------------------------------------------------------------------
NEWSDATA_BASE = "https://newsdata.io/api/1/latest"


def _search_newsdata(query: str, key: str, max_results: int, language: str) -> List[Dict]:
    # Use spaCy-aware entity extraction when available, then ASCII-sanitise
    try:
        from src.preprocessing.claim_extractor import claim_to_query
        entity_query = claim_to_query(query, max_words=8)
        safe_query = make_search_query(entity_query or query, max_words=8, max_chars=_NEWSDATA_MAX_Q_CHARS)
    except Exception:
        safe_query = make_search_query(query, max_words=8, max_chars=_NEWSDATA_MAX_Q_CHARS)
    if not safe_query:
        logger.warning("NewsData: query sanitised to empty string — skipping.")
        return []

    params = {
        "apikey":   key,
        "q":        safe_query,
        "language": language,
    }
    logger.debug("NewsData query: '%s' (%d chars)", safe_query, len(safe_query))
    try:
        resp = requests.get(NEWSDATA_BASE, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            logger.error("NewsData error: %s", data.get("message"))
            return []
        return _normalize_newsdata((data.get("results") or [])[:max_results])
    except requests.RequestException as exc:
        logger.error("NewsData request failed: %s", exc)
        return []


def _normalize_newsdata(articles: list) -> List[Dict]:
    return [
        {
            "title":       a.get("title") or "",
            "description": a.get("description") or "",
            "url":         a.get("link") or "",
            "source":      a.get("source_id") or a.get("source_name") or "Unknown",
            "publishedAt": a.get("pubDate") or "",
            "content":     a.get("content") or "",
        }
        for a in articles
    ]


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------
_CACHE_TTL  = 600   # 10 minutes
_CACHE_MAX  = 128
_cache: Dict[str, tuple] = {}
_cache_lock = threading.Lock()


def _cache_key(query: str, api_key: str, max_results: int, language: str) -> str:
    raw = f"{query}|{api_key or ''}|{max_results}|{language}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(key: str) -> Optional[List[Dict]]:
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        ts, results = entry
        if time.time() - ts > _CACHE_TTL:
            del _cache[key]
            return None
        return results


def _cache_put(key: str, results: List[Dict]) -> None:
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            oldest = min(_cache, key=lambda k: _cache[k][0])
            del _cache[oldest]
        _cache[key] = (time.time(), results)


# ---------------------------------------------------------------------------
# Public unified entry point
# ---------------------------------------------------------------------------

def search_news(
    query: str,
    api_key: Optional[str] = None,
    max_results: int = 5,
    language: str = "en",
    sort_by: str = "relevancy",
) -> List[Dict[str, Any]]:
    """
    Search for news articles matching *query*.

    The raw *query* (which may be a full sentence with special characters)
    is automatically sanitised into a short, API-safe keyword phrase via
    make_search_query() before being sent to any provider.

    Provider selection:
      - key starts with 'pub_' → NewsData.io
      - 32-char hex key        → NewsAPI.org
      - no key                 → Google News RSS + DuckDuckGo (free)

    Fallback chain:
      paid API → free sources (if paid returns 0 results or fails)

    Results are cached for 10 minutes to avoid redundant API calls.
    Never raises — returns [] if all sources fail.
    """
    from src.retrieval.free_news_search import search_news_free

    raw_key = (
        api_key
        or os.environ.get("NEWSAPI_KEY", "")
        or os.environ.get("NEWSDATA_KEY", "")
    )

    # Cache key uses the RAW query so identical sentences hit the same cache slot
    ck = _cache_key(query, raw_key, max_results, language)
    cached = _cache_get(ck)
    if cached is not None:
        logger.info("Cache hit for query: '%s'", query[:60])
        return cached

    # No API key → free sources directly
    if not raw_key or not raw_key.strip():
        logger.info("No API key — using free sources.")
        results = search_news_free(query, max_results=max_results, language=language)
        _cache_put(ck, results)
        return results

    # Paid provider
    key      = _clean_key(raw_key)
    provider = detect_provider(key)
    logger.info("Provider: %s | raw query length: %d chars", provider, len(query))

    results = (
        _search_newsdata(query, key, max_results, language)
        if provider == "newsdata"
        else _search_newsapi(query, key, max_results, language)
    )

    # Fallback to free if paid returned nothing
    if not results:
        logger.info("Paid API returned 0 results — falling back to free sources.")
        results = search_news_free(query, max_results=max_results, language=language)

    _cache_put(ck, results)
    return results
