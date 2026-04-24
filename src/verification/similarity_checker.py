#!/usr/bin/env python3
"""
similarity_checker.py
---------------------
Compute semantic similarity between two texts using:
  - Sentence-BERT (preferred, via sentence-transformers)
  - TF-IDF cosine similarity (fast fallback, no GPU needed)

The SBERT model is lazy-loaded on first use to avoid blocking
app startup when similarity isn't needed (e.g. Quick Predict).

Fix applied (issue #4):
  The old _tfidf_similarity() fitted a TfidfVectorizer on only 2 documents,
  making IDF scores meaningless (every word has IDF=0 or log(2)).
  similarity_score() now delegates to batch_similarity([text_b]) instead,
  so the same reliable code path is used whether comparing one or many texts.
"""
from __future__ import annotations
import logging
from typing import List

logger = logging.getLogger(__name__)

__all__ = ["similarity_score", "best_similarity", "batch_similarity"]

# ---------------------------------------------------------------------------
# sentence-transformers — check availability at import, load model on demand
# ---------------------------------------------------------------------------
_SBERT_MODEL = None
_SBERT_AVAILABLE = False

try:
    import sentence_transformers  # noqa: F401 — availability check only
    _SBERT_AVAILABLE = True
except ImportError:
    logger.info("sentence-transformers not installed. Will use TF-IDF fallback.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
import numpy as np


def _get_sbert_model():
    """Lazy-load the Sentence-BERT model on first use."""
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading Sentence-BERT model: all-MiniLM-L6-v2 ...")
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Sentence-BERT model loaded successfully.")
    return _SBERT_MODEL


# ---------------------------------------------------------------------------
# Batch similarity — one query vs many candidates in a single pass
# This is the ONLY place where actual similarity computation happens.
# All single-pair calls route through here to guarantee consistent IDF context.
# ---------------------------------------------------------------------------

def _sbert_batch_similarity(query: str, candidates: List[str]) -> List[float]:
    """Encode query + all candidates in one batch, then compute cosine."""
    from sentence_transformers import util as st_util
    model = _get_sbert_model()
    emb_q = model.encode(query, convert_to_tensor=True)
    emb_c = model.encode(candidates, convert_to_tensor=True)
    scores = st_util.cos_sim(emb_q, emb_c)[0]  # 1×N tensor
    return [max(0.0, min(1.0, float(s))) for s in scores]


def _tfidf_batch_similarity(query: str, candidates: List[str]) -> List[float]:
    """
    Fit ONE TfidfVectorizer on query + ALL candidates together, then compute
    cosine similarity of query (row 0) vs each candidate (rows 1..N).

    Having N+1 documents in the corpus gives IDF scores meaningful context —
    words that appear in many candidates are downweighted, distinguishing
    words are upweighted. This is why all pairwise comparisons must go through
    the batch path and never fit a 2-document vectorizer.
    """
    try:
        texts = [query] + candidates
        vec = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = vec.fit_transform(texts)
        scores = _cos_sim(tfidf[0:1], tfidf[1:]).flatten()
        return [max(0.0, min(1.0, float(s))) for s in scores]
    except Exception:
        return [0.0] * len(candidates)


def batch_similarity(query: str, candidates: List[str]) -> List[float]:
    """
    Compute similarity of *query* against each text in *candidates* in a
    single batch pass, returning a list of scores in [0, 1].

    This is the canonical similarity function — all other similarity
    functions in this module delegate here.

    Returns a list of 0.0s if candidates is empty or query is empty.
    """
    if not query or not candidates:
        return [0.0] * len(candidates)

    # Filter out empty strings but remember original positions
    valid = [(i, c) for i, c in enumerate(candidates) if c and c.strip()]
    if not valid:
        return [0.0] * len(candidates)

    indices, valid_texts = zip(*valid)

    if _SBERT_AVAILABLE:
        try:
            valid_scores = _sbert_batch_similarity(query, list(valid_texts))
        except Exception as exc:
            logger.warning("SBERT batch failed (%s). Falling back to TF-IDF.", exc)
            valid_scores = _tfidf_batch_similarity(query, list(valid_texts))
    else:
        valid_scores = _tfidf_batch_similarity(query, list(valid_texts))

    # Map scores back to original positions (empty candidates stay 0.0)
    result = [0.0] * len(candidates)
    for idx, score in zip(indices, valid_scores):
        result[idx] = score
    return result


# ---------------------------------------------------------------------------
# Single-pair API — delegates to batch_similarity for correct IDF context
# ---------------------------------------------------------------------------

def similarity_score(text_a: str, text_b: str) -> float:
    """
    Return cosine similarity in [0, 1] between *text_a* and *text_b*.

    Delegates to batch_similarity([text_b]) so TF-IDF IDF scores are
    computed with a proper corpus context (not a degenerate 2-doc corpus).
    Uses Sentence-BERT if available, otherwise TF-IDF.
    """
    if not text_a or not text_b:
        return 0.0
    scores = batch_similarity(text_a, [text_b])
    return scores[0] if scores else 0.0


def best_similarity(query: str, candidates: List[str]) -> float:
    """
    Return the highest similarity score between *query* and any text in
    *candidates*. Returns 0.0 if candidates is empty.
    """
    if not candidates:
        return 0.0
    scores = batch_similarity(query, candidates)
    return float(np.max(scores)) if scores else 0.0
