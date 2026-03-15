#!/usr/bin/env python3
"""
web_scraper.py
--------------
Extract full article text from a URL using newspaper3k (preferred)
or a BeautifulSoup fallback.
"""
from __future__ import annotations
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["extract_article_text"]

# ---------------------------------------------------------------------------
# newspaper3k is optional
# ---------------------------------------------------------------------------
try:
    from newspaper import Article as _Article
    _USE_NEWSPAPER = True
except ImportError:
    _USE_NEWSPAPER = False

try:
    import requests
    from bs4 import BeautifulSoup
    _USE_BS4 = True
except ImportError:
    _USE_BS4 = False


def _newspaper_extract(url: str) -> Optional[str]:
    try:
        article = _Article(url)
        article.download()
        article.parse()
        return article.text.strip() or None
    except Exception as exc:
        logger.debug("newspaper3k failed for %s: %s", url, exc)
        return None


def _bs4_extract(url: str, timeout: int = 10) -> Optional[str]:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove scripts / styles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text or None
    except Exception as exc:
        logger.debug("bs4 scrape failed for %s: %s", url, exc)
        return None


def extract_article_text(url: str) -> str:
    """
    Download and return the main text content of *url*.

    Tries newspaper3k first, falls back to BeautifulSoup.
    Returns an empty string if both fail.
    """
    if not url:
        return ""

    if _USE_NEWSPAPER:
        text = _newspaper_extract(url)
        if text:
            return text

    if _USE_BS4:
        text = _bs4_extract(url)
        if text:
            return text

    logger.warning("Could not extract text from %s (both scrapers failed).", url)
    return ""
