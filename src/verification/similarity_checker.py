#!/usr/bin/env python3
"""
similarity_checker.py
---------------------
Compute semantic similarity between two texts using:
  - Sentence-BERT (preferred, via sentence-transformers)
  - TF-IDF cosine similarity (fast fallback, no GPU needed)
"""
from __future__ import annotations
import logging
from typing import List

logger = logging.getLogger(__name__)

__all__ = ["similarity_score", "best_similarity"]

# ---------------------------------------------------------------------------
# sentence-transformers is optional
# ---------------------------------------------------------------------------
_SBERT_MODEL = None

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Loaded Sentence-BERT model: all-MiniLM-L6-v2")
    _USE_SBERT = True
except Exception as exc:
    logger.warning("sentence-transformers not available (%s). Using TF-IDF fallback.", exc)
    _USE_SBERT = False

# TF-IDF fallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
import numpy as np


def _sbert_similarity(text_a: str, text_b: str) -> float:
    emb_a = _SBERT_MODEL.encode(text_a, convert_to_tensor=True)
    emb_b = _SBERT_MODEL.encode(text_b, convert_to_tensor=True)
    score = float(st_util.cos_sim(emb_a, emb_b))
    return max(0.0, min(1.0, score))


def _tfidf_similarity(text_a: str, text_b: str) -> float:
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = vec.fit_transform([text_a, text_b])
        score = float(_cos_sim(tfidf[0], tfidf[1])[0, 0])
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def similarity_score(text_a: str, text_b: str) -> float:
    """
    Return cosine similarity in [0, 1] between *text_a* and *text_b*.
    Uses Sentence-BERT if available, otherwise TF-IDF.
    """
    if not text_a or not text_b:
        return 0.0
    if _USE_SBERT:
        return _sbert_similarity(text_a, text_b)
    return _tfidf_similarity(text_a, text_b)


def best_similarity(query: str, candidates: List[str]) -> float:
    """
    Return the highest similarity score between *query* and any text in *candidates*.
    Returns 0.0 if candidates is empty.
    """
    if not candidates:
        return 0.0
    scores = [similarity_score(query, c) for c in candidates if c]
    return float(np.max(scores)) if scores else 0.0
