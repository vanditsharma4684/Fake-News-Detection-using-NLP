#!/usr/bin/env python3
"""
Lightweight text normalization for the fake-news verifier.
"""
from __future__ import annotations
import re
from typing import List, Optional, Sequence

URL_RE    = re.compile(r"https?://\S+")
EMAIL_RE  = re.compile(r"\S+@\S+")
ASCII_RE  = re.compile(r"[^\x00-\x7F]+")
SPACE_RE  = re.compile(r"\s+")

# Patterns that leak the label in the Kaggle True/Fake news dataset
# e.g. "WASHINGTON (Reuters) - ", "(AP) — ", "21st Century Wire says"
_SOURCE_MARKERS = [
    re.compile(r"\(Reuters\)\s*[-–—]?\s*", re.IGNORECASE),
    re.compile(r"\(AP\)\s*[-–—]?\s*", re.IGNORECASE),
    re.compile(r"\(AFP\)\s*[-–—]?\s*", re.IGNORECASE),
    # City-name datelines: "WASHINGTON (Reuters) -", "NEW YORK (AP) -"
    re.compile(r"^[A-Z][A-Z\s,]{2,30}\((?:Reuters|AP|AFP)\)\s*[-–—]\s*", re.MULTILINE),
    # "21st Century Wire" and similar fake-news source stamps
    re.compile(r"21st\s+century\s+wire\s+says?\s*", re.IGNORECASE),
]

__all__ = ["clean_text", "clean_many", "strip_source_markers"]


def strip_source_markers(text: str) -> str:
    """
    Remove news-agency attribution patterns that can leak labels
    during training (e.g. Reuters/AP bylines only appear in REAL articles).
    """
    if not isinstance(text, str):
        return ""
    s = text
    for pat in _SOURCE_MARKERS:
        s = pat.sub("", s)
    return s.strip()


def clean_text(
    text: Optional[str],
    *,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_non_ascii: bool = True,
    collapse_whitespace: bool = True,
    strip_sources: bool = False,
) -> str:
    if not isinstance(text, str):
        return ""
    s = text
    if strip_sources:      s = strip_source_markers(s)
    if lowercase:          s = s.lower()
    if remove_urls:        s = URL_RE.sub(" ", s)
    if remove_emails:      s = EMAIL_RE.sub(" ", s)
    if remove_non_ascii:   s = ASCII_RE.sub(" ", s)
    if collapse_whitespace: s = SPACE_RE.sub(" ", s).strip()
    return s


def clean_many(texts: Sequence[Optional[str]], **kwargs) -> List[str]:
    return [clean_text(t, **kwargs) for t in texts]

