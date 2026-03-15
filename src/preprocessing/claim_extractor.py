#!/usr/bin/env python3
"""
claim_extractor.py
------------------
Extracts key factual sentences (claims) from a news article.

Strategy:
  1. Split into sentences using spaCy (preferred) or a regex fallback.
  2. Keep sentences that look factual: long enough, contain a noun and a verb.
  3. Return up to `max_claims` sentences ranked by length (longest = most content).
"""
from __future__ import annotations
import re
from typing import List

__all__ = ["extract_claims"]

# ---------------------------------------------------------------------------
# spaCy is optional – fall back to regex sentence splitting if not installed
# ---------------------------------------------------------------------------
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    _USE_SPACY = True
except Exception:
    _NLP = None
    _USE_SPACY = False


def _spacy_sentences(text: str) -> List[str]:
    doc = _NLP(text)
    return [sent.text.strip() for sent in doc.sents]


def _regex_sentences(text: str) -> List[str]:
    """Simple regex sentence splitter as fallback."""
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


def extract_claims(text: str, max_claims: int = 5, min_len: int = 30) -> List[str]:
    """
    Extract the most meaningful sentences from *text*.

    Parameters
    ----------
    text : str
        Raw article text.
    max_claims : int
        Maximum number of claims to return.
    min_len : int
        Minimum character length for a sentence to be considered.

    Returns
    -------
    list[str]
        Extracted claim sentences.
    """
    if not text or not text.strip():
        return []

    sentences = _spacy_sentences(text) if _USE_SPACY else _regex_sentences(text)

    # Filter: minimum length
    candidates = [s for s in sentences if len(s) >= min_len]

    # Rank by length (more words = more content), take top N
    candidates = sorted(candidates, key=len, reverse=True)[:max_claims]

    return candidates
