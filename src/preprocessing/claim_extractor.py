#!/usr/bin/env python3
"""
claim_extractor.py
------------------
Extracts key factual sentences (claims) from a news article, and converts
them into tight keyword queries for news search APIs.

Two public functions:

  extract_claims(text, max_claims, min_len)
      Splits the article into sentences and returns the most informative
      ones ranked by named-entity richness + verb strength (not length).

  claim_to_query(claim, max_words)
      Converts a full claim sentence into a short (<=8 word), ASCII-only
      keyword phrase suitable for NewsData.io / NewsAPI.org queries.
      Uses spaCy NER when available; falls back to simple word extraction.
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import List, Tuple

__all__ = ["extract_claims", "claim_to_query"]

# ---------------------------------------------------------------------------
# spaCy - optional; regex fallback used if not installed
# ---------------------------------------------------------------------------
try:
    import spacy as _spacy
    _NLP = _spacy.load("en_core_web_sm")
    _USE_SPACY = True
except Exception:
    _NLP = None
    _USE_SPACY = False

# Named entity labels that carry strong factual signal
_FACTUAL_ENT_LABELS = {
    "ORG", "PERSON", "GPE", "LOC", "EVENT",
    "PRODUCT", "LAW", "NORP", "FAC", "WORK_OF_ART",
}

# Auxiliary / modal verb lemmas excluded from verb scoring
_AUX_LEMMAS = {
    "be", "have", "do", "will", "would", "could", "should",
    "may", "might", "can", "shall", "must", "need", "dare",
    "is", "are", "was", "were", "am", "been", "being",
}

_AUX_TAGS = {"MD", "VBZ", "VBP"}


# ---------------------------------------------------------------------------
# spaCy-based sentence scoring
# ---------------------------------------------------------------------------

def _score_spacy(sent) -> float:
    text = sent.text
    ent_score = sum(
        2.0 for token in sent
        if token.ent_type_ in _FACTUAL_ENT_LABELS and token.ent_iob_ == "B"
    )
    verb_score = sum(
        1.0 for token in sent
        if token.pos_ == "VERB"
        and token.tag_ not in _AUX_TAGS
        and token.lemma_.lower() not in _AUX_LEMMAS
    )
    num_score = sum(
        0.5 for token in sent
        if token.like_num or token.ent_type_ in {
            "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL",
        }
    )
    length_bonus = math.log1p(len(text)) * 0.1
    return ent_score + verb_score + num_score + length_bonus


def _spacy_extract(text: str, max_claims: int, min_len: int) -> List[str]:
    doc = _NLP(text)
    candidates: List[Tuple[float, str]] = []
    for sent in doc.sents:
        s = sent.text.strip()
        if len(s) >= min_len:
            candidates.append((_score_spacy(sent), s))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in candidates[:max_claims]]


# ---------------------------------------------------------------------------
# Regex-based sentence scoring (no spaCy)
# ---------------------------------------------------------------------------

_REGEX_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PROPER_NOUN_RE   = re.compile(r"\b(?:[A-Z][a-z]{1,}|[A-Z]{2,})\b")
_NUMBER_RE        = re.compile(r"\b\d[\d,\.]*\b")
_STRONG_VERB_RE   = re.compile(
    r"\b(?:confirm|announce|reveal|claim|say|state|warn|deny|accuse|"
    r"arrest|kill|injure|launch|sign|pass|vote|elect|appoint|resign|"
    r"discover|find|report|allege|sue|ban|approve|reject|call|urge|"
    r"demand|propose|declare|threaten|attack|win|lose|hit|strike|"
    r"crash|explode|collapse|flood|burn|shoot|fire)\w*\b",
    re.IGNORECASE,
)


def _score_regex(sentence: str) -> float:
    proper_nouns = len(_PROPER_NOUN_RE.findall(sentence))
    numbers      = len(_NUMBER_RE.findall(sentence))
    strong_verbs = len(_STRONG_VERB_RE.findall(sentence))
    length_bonus = math.log1p(len(sentence)) * 0.1
    return proper_nouns * 1.5 + numbers * 1.0 + strong_verbs * 2.0 + length_bonus


def _regex_extract(text: str, max_claims: int, min_len: int) -> List[str]:
    raw = _REGEX_SENT_SPLIT.split(text)
    candidates: List[Tuple[float, str]] = []
    for s in raw:
        s = s.strip()
        if len(s) >= min_len:
            candidates.append((_score_regex(s), s))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in candidates[:max_claims]]


# ---------------------------------------------------------------------------
# Public function 1 - extract_claims
# ---------------------------------------------------------------------------

def extract_claims(text: str, max_claims: int = 5, min_len: int = 30) -> List[str]:
    """
    Extract the most newsworthy / factual sentences from *text*.

    Ranks by named-entity richness + strong verb count + numbers.
    A short "NASA confirmed asteroid will hit Earth" outscores a long
    background-context sentence.

    Parameters
    ----------
    text : str
        Raw article text or headline.
    max_claims : int
        Maximum number of claims to return (default 5).
    min_len : int
        Minimum character length for a sentence to qualify (default 30).

    Returns
    -------
    list[str]
        Up to *max_claims* sentences, best first.
    """
    if not text or not text.strip():
        return []
    if _USE_SPACY:
        return _spacy_extract(text, max_claims, min_len)
    return _regex_extract(text, max_claims, min_len)


# ---------------------------------------------------------------------------
# Public function 2 - claim_to_query
# ---------------------------------------------------------------------------

_QUERY_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "that", "this", "these",
    "those", "it", "its", "said", "says", "saying", "told", "tell", "tells",
    "night", "morning", "day", "according", "after", "also", "about",
    "saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday",
}


def _ascii_clean(text: str) -> str:
    """Normalise Unicode to ASCII, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def claim_to_query(claim: str, max_words: int = 8) -> str:
    """
    Convert a full claim sentence into a short keyword query.

    Output is guaranteed: ASCII-only, <= max_words words, <= 100 chars.

    With spaCy: extracts named entities first, then numbers, then strong
    verbs, then nouns — in that priority order.

    Without spaCy: strips stopwords, takes first max_words content words.

    Parameters
    ----------
    claim : str
        Raw claim sentence (may contain Unicode and punctuation).
    max_words : int
        Maximum words in the output (default 8).

    Returns
    -------
    str
        Clean keyword phrase.
    """
    if not claim or not claim.strip():
        return ""

    if not _USE_SPACY:
        clean = _ascii_clean(claim)
        words = [w for w in clean.split() if w.lower() not in _QUERY_STOPWORDS and len(w) > 1]
        return " ".join(words[:max_words])[:100]

    doc = _NLP(claim)
    seen:   set  = set()
    tokens: list = []

    # Pass 1 — named entities (highest signal)
    for ent in doc.ents:
        text = ent.text.strip()
        key  = text.lower()
        if key not in seen and len(text) > 1:
            seen.add(key)
            tokens.append(text)

    # Pass 2 — standalone numbers / quantities
    for token in doc:
        if token.like_num or token.ent_type_ in {
            "CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY"
        }:
            key = token.text.lower()
            if key not in seen and len(token.text) > 1:
                seen.add(key)
                tokens.append(token.text)

    # Pass 3 — strong non-auxiliary verbs
    for token in doc:
        if (token.pos_ == "VERB"
                and token.lemma_.lower() not in _AUX_LEMMAS
                and token.text.lower() not in seen
                and len(token.text) > 2):
            seen.add(token.text.lower())
            tokens.append(token.text)

    # Pass 4 — nouns to fill remaining budget
    for token in doc:
        if (token.pos_ in {"NOUN", "PROPN"}
                and token.text.lower() not in seen
                and not token.is_stop
                and len(token.text) > 2):
            seen.add(token.text.lower())
            tokens.append(token.text)

    result = _ascii_clean(" ".join(tokens[:max_words]))

    if not result:
        result = _ascii_clean(" ".join(claim.split()[:max_words]))

    return result[:100]
