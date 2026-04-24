#!/usr/bin/env python3
"""
free_news_search.py
-------------------
Fetch news articles from FREE sources that require no API key:

  1. Google News RSS feed (primary)  — fast, reliable, no rate limits
  2. DuckDuckGo HTML search (fallback) — scrapes news results

Changes applied:
  - _sanitise_query() added: cleans raw claim sentences before sending to
    Google News RSS (removes Unicode curly quotes, em-dashes, long sentences).
  - credibility_url field added to every article dict: holds the real
    publisher domain from the RSS <source url="..."> attribute, NOT the
    news.google.com redirect URL. The credibility scorer uses this field
    so Google RSS articles get proper scores (0.95 for Reuters, not 0.35).
  - DuckDuckGo queries also sanitised via _sanitise_query().
"""
from __future__ import annotations

import logging
import re
import unicodedata
import xml.etree.ElementTree as ET
from html import unescape
from typing import Any, Dict, List
from urllib.parse import quote_plus, urlparse

import requests

logger = logging.getLogger(__name__)

__all__ = ["search_news_free"]

DEFAULT_TIMEOUT = 15
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}


# ---------------------------------------------------------------------------
# Query sanitiser
# ---------------------------------------------------------------------------

def _sanitise_query(query: str, max_words: int = 10) -> str:
    """
    Convert a raw claim sentence into a clean keyword query.

    - Normalises Unicode to ASCII (curly quotes, em-dashes, accents)
    - Strips punctuation
    - Takes first max_words words
    """
    normalised = unicodedata.normalize("NFKD", query)
    ascii_only  = normalised.encode("ascii", errors="ignore").decode("ascii")
    clean = re.sub(r"[^\w\s]", " ", ascii_only)
    clean = re.sub(r"\s+", " ", clean).strip()
    return " ".join(clean.split()[:max_words])


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _domain_from_url(url: str) -> str:
    """Extract a clean domain name from a URL."""
    try:
        parsed = urlparse(url)
        host = parsed.netloc or parsed.path.split("/")[0]
        return re.sub(r"^www\.", "", host)
    except Exception:
        return "Unknown"


def _extract_ddg_url(ddg_href: str) -> str:
    """
    DuckDuckGo wraps external URLs in redirect links.
    Extract the real destination URL.
    """
    m = re.search(r"uddg=([^&]+)", ddg_href)
    if m:
        from urllib.parse import unquote
        return unquote(m.group(1))
    if ddg_href.startswith("http"):
        return ddg_href
    return ddg_href


# ---------------------------------------------------------------------------
# 1. Google News RSS  (no key, no rate-limit, XML response)
# ---------------------------------------------------------------------------
_GNEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def _search_google_news_rss(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch articles from Google News RSS feed.

    Important: Google News RSS links are redirect URLs (news.google.com/articles/...).
    We expose the real publisher URL via 'credibility_url' using the
    <source url="..."> attribute so the credibility scorer can rate sources
    correctly (Reuters = 0.97, not 0.35 for news.google.com).
    """
    safe_query = _sanitise_query(query, max_words=10)
    url = _GNEWS_RSS.format(query=quote_plus(safe_query))

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Google News RSS request failed: %s", exc)
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        logger.warning("Google News RSS XML parse error: %s", exc)
        return []

    articles: List[Dict[str, Any]] = []

    for item in root.iter("item"):
        title_el   = item.find("title")
        link_el    = item.find("link")
        pubdate_el = item.find("pubDate")
        desc_el    = item.find("description")
        source_el  = item.find("source")

        title       = unescape(title_el.text.strip())   if title_el   is not None and title_el.text   else ""
        link        = link_el.text.strip()               if link_el    is not None and link_el.text    else ""
        pubdate     = pubdate_el.text.strip()            if pubdate_el is not None and pubdate_el.text else ""
        desc_raw    = unescape(desc_el.text.strip())     if desc_el    is not None and desc_el.text    else ""
        source_name = source_el.text.strip()             if source_el  is not None and source_el.text  else ""
        # This is the REAL publisher URL (e.g. "https://www.reuters.com")
        source_url  = source_el.get("url", "")          if source_el  is not None                     else ""

        description = re.sub(r"<[^>]+>", "", desc_raw).strip()

        # Strip "- SourceName" suffix Google appends to titles
        clean_title = title
        if source_name and title.endswith(f" - {source_name}"):
            clean_title = title[: -len(f" - {source_name}")].strip()

        articles.append({
            "title":           clean_title,
            "description":     description,
            # 'url' is the Google redirect — do NOT use for credibility scoring
            "url":             link,
            # 'credibility_url' is the real publisher domain — USE THIS for scoring
            "credibility_url": source_url or link,
            "source":          source_name or _domain_from_url(source_url or link),
            "publishedAt":     pubdate,
            "content":         description,
        })

        if len(articles) >= max_results:
            break

    logger.info("Google News RSS returned %d results for '%s'", len(articles), safe_query[:60])
    return articles


# ---------------------------------------------------------------------------
# 2. DuckDuckGo HTML news search (no key, no JS required)
# ---------------------------------------------------------------------------
_DDG_NEWS_URL = "https://html.duckduckgo.com/html/"


def _search_duckduckgo_news(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Scrape news results from DuckDuckGo HTML-only interface.
    Lightweight fallback — no JS / API key needed.
    """
    safe_query = _sanitise_query(query, max_words=10)

    try:
        resp = requests.post(
            _DDG_NEWS_URL,
            data={"q": f"{safe_query} news", "kl": "us-en"},
            headers=_HEADERS,
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("DuckDuckGo request failed: %s", exc)
        return []

    articles: List[Dict[str, Any]] = []

    try:
        from bs4 import BeautifulSoup
        soup     = BeautifulSoup(resp.text, "html.parser")
        results  = soup.select(".result__a")
        snippets = soup.select(".result__snippet")

        for i, link_tag in enumerate(results[:max_results]):
            href     = link_tag.get("href", "")
            title    = link_tag.get_text(strip=True)
            snippet  = snippets[i].get_text(strip=True) if i < len(snippets) else ""
            real_url = _extract_ddg_url(href)

            articles.append({
                "title":           title,
                "description":     snippet,
                "url":             real_url,
                "credibility_url": real_url,
                "source":          _domain_from_url(real_url),
                "publishedAt":     "",
                "content":         snippet,
            })

    except ImportError:
        # Regex fallback if BS4 not installed
        pattern = re.compile(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.+?)</a>', re.DOTALL
        )
        for match in pattern.finditer(resp.text):
            href, title_raw = match.group(1), match.group(2)
            title    = re.sub(r"<[^>]+>", "", title_raw).strip()
            real_url = _extract_ddg_url(href)
            articles.append({
                "title":           title,
                "description":     "",
                "url":             real_url,
                "credibility_url": real_url,
                "source":          _domain_from_url(real_url),
                "publishedAt":     "",
                "content":         "",
            })
            if len(articles) >= max_results:
                break

    logger.info("DuckDuckGo returned %d results for '%s'", len(articles), safe_query[:60])
    return articles


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def search_news_free(
    query: str,
    max_results: int = 5,
    language: str = "en",
) -> List[Dict[str, Any]]:
    """
    Search for news articles matching *query* using FREE sources.
    No API key required.

    Tries:
      1. Google News RSS feed (fast, reliable)
      2. DuckDuckGo HTML search (supplement / fallback)

    Returns
    -------
    list[dict]
        Normalised article dicts with keys:
        title, description, url, credibility_url, source, publishedAt, content
    """
    results = _search_google_news_rss(query, max_results=max_results)

    if len(results) >= max_results:
        return results[:max_results]

    needed = max_results - len(results)
    if needed > 0:
        ddg_results = _search_duckduckgo_news(query, max_results=needed)
        seen_urls = {a["url"] for a in results}
        for a in ddg_results:
            if a["url"] not in seen_urls:
                results.append(a)
                seen_urls.add(a["url"])

    return results[:max_results]
