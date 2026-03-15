#!/usr/bin/env python3
"""
credibility_scorer.py
---------------------
Assign a credibility score [0, 1] to a news source domain.

The trusted-sources dictionary is curated manually. Unknown domains
receive a conservative default score (0.35).
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

__all__ = ["credibility_score", "average_credibility"]

# ---------------------------------------------------------------------------
# Trusted source database  (domain → credibility score)
# Tier 1 (0.90–0.98): Major wire services, public broadcasters
# Tier 2 (0.80–0.89): Large reputable outlets
# Tier 3 (0.65–0.79): Generally reliable but editorial slant possible
# Unknown: 0.35
# ---------------------------------------------------------------------------
TRUSTED_SOURCES: Dict[str, float] = {
    # Wire services
    "reuters.com": 0.97,
    "apnews.com": 0.97,
    "afp.com": 0.95,
    # Public / national broadcasters
    "bbc.com": 0.95,
    "bbc.co.uk": 0.95,
    "npr.org": 0.93,
    "pbs.org": 0.93,
    "abc.net.au": 0.90,
    # Mainstream US outlets
    "nytimes.com": 0.90,
    "washingtonpost.com": 0.89,
    "wsj.com": 0.88,
    "usatoday.com": 0.82,
    "cnn.com": 0.82,
    "nbcnews.com": 0.82,
    "cbsnews.com": 0.82,
    "abcnews.go.com": 0.82,
    "msnbc.com": 0.78,
    "foxnews.com": 0.72,
    # International
    "theguardian.com": 0.88,
    "theatlantic.com": 0.85,
    "economist.com": 0.90,
    "ft.com": 0.90,
    "dw.com": 0.90,
    "aljazeera.com": 0.80,
    "france24.com": 0.82,
    # Science / fact-check
    "snopes.com": 0.90,
    "factcheck.org": 0.92,
    "politifact.com": 0.90,
    "sciencemag.org": 0.95,
    "nature.com": 0.97,
    "who.int": 0.97,
    "cdc.gov": 0.97,
    "nih.gov": 0.97,
    "gov.uk": 0.93,
    # Indian outlets
    "thehindu.com": 0.88,
    "hindustantimes.com": 0.80,
    "ndtv.com": 0.80,
    "timesofindia.indiatimes.com": 0.78,
    "indianexpress.com": 0.82,
    "pib.gov.in": 0.95,
}

DEFAULT_SCORE = 0.35  # unknown / unrated domain


def _extract_domain(url_or_name: str) -> str:
    """Extract bare domain from a URL or source name string."""
    s = url_or_name.strip().lower()
    # If it looks like a URL, parse it
    if s.startswith("http"):
        try:
            parsed = urlparse(s)
            host = parsed.netloc or parsed.path
        except Exception:
            host = s
    else:
        host = s
    # Remove www. prefix
    host = re.sub(r"^www\.", "", host)
    # Strip path
    host = host.split("/")[0].strip()
    return host


def credibility_score(url_or_name: str) -> float:
    """
    Return credibility score for *url_or_name* in [0, 1].

    Parameters
    ----------
    url_or_name : str
        Full URL ("https://bbc.com/news/...") or domain/source name ("BBC").

    Returns
    -------
    float
        Credibility score.
    """
    if not url_or_name:
        return DEFAULT_SCORE
    domain = _extract_domain(url_or_name)
    # Exact match
    if domain in TRUSTED_SOURCES:
        return TRUSTED_SOURCES[domain]
    # Partial match (e.g. "news.bbc.co.uk" → "bbc.co.uk")
    for known_domain, score in TRUSTED_SOURCES.items():
        if domain.endswith(known_domain) or known_domain.endswith(domain):
            return score
    return DEFAULT_SCORE


def average_credibility(urls_or_names: List[str]) -> float:
    """Return the mean credibility of a list of sources (ignores empty strings)."""
    filtered = [u for u in urls_or_names if u]
    if not filtered:
        return DEFAULT_SCORE
    scores = [credibility_score(u) for u in filtered]
    return sum(scores) / len(scores)
