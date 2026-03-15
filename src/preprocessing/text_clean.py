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

__all__ = ["clean_text", "clean_many"]


def clean_text(
    text: Optional[str],
    *,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_non_ascii: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    if not isinstance(text, str):
        return ""
    s = text
    if lowercase:          s = s.lower()
    if remove_urls:        s = URL_RE.sub(" ", s)
    if remove_emails:      s = EMAIL_RE.sub(" ", s)
    if remove_non_ascii:   s = ASCII_RE.sub(" ", s)
    if collapse_whitespace: s = SPACE_RE.sub(" ", s).strip()
    return s


def clean_many(texts: Sequence[Optional[str]], **kwargs) -> List[str]:
    return [clean_text(t, **kwargs) for t in texts]
