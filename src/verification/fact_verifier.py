#!/usr/bin/env python3
"""
fact_verifier.py
----------------
Orchestrates all components into a single verification pipeline.

Pipeline:
  1. Clean text
  2. Extract claims
  3. Search NewsAPI for each claim
  4. (Optionally) scrape full article text
  5. Compute semantic similarity between input and retrieved articles
  6. Score source credibility
  7. Run ML fake-news classifier
  8. Combine into a final credibility score [0, 1]

Usage
-----
    from src.verification.fact_verifier import FactVerifier

    verifier = FactVerifier(
        pipeline_path="models/pipeline.joblib",
        newsapi_key="YOUR_KEY",
    )
    result = verifier.verify("NASA confirmed a massive asteroid will hit Earth.")
    print(result)
"""
from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["FactVerifier", "VerificationResult"]


@dataclass
class VerificationResult:
    label: str                          # "REAL" or "FAKE"
    credibility_score: float            # 0.0 – 1.0  (higher = more credible)
    ml_fake_probability: float          # Raw ML model output
    source_similarity: float            # Avg similarity with retrieved articles
    source_credibility: float           # Avg credibility of retrieved sources
    supporting_sources: List[Dict]      # Articles found
    claims_extracted: List[str]         # Sentences sent to search
    threshold: float                    # Decision threshold used
    explanation: str = ""               # Human-readable summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "credibility_score": round(self.credibility_score, 4),
            "ml_fake_probability": round(self.ml_fake_probability, 4),
            "source_similarity": round(self.source_similarity, 4),
            "source_credibility": round(self.source_credibility, 4),
            "supporting_sources": self.supporting_sources,
            "claims_extracted": self.claims_extracted,
            "threshold": self.threshold,
            "explanation": self.explanation,
        }


class FactVerifier:
    """
    Hybrid fake-news + fact-verification system.

    Parameters
    ----------
    pipeline_path : str | Path
        Path to the trained scikit-learn pipeline (.joblib).
        Falls back to model_path + vectorizer_path if not given.
    model_path : str | Path, optional
    vectorizer_path : str | Path, optional
    newsapi_key : str, optional
        NewsAPI key. Reads ``NEWSAPI_KEY`` env var if omitted.
    threshold : float
        Credibility score below this → FAKE. Default 0.45.
    w_ml : float
        Weight of ML classifier in final score.
    w_sim : float
        Weight of source similarity.
    w_cred : float
        Weight of source credibility.
    max_sources : int
        How many articles to retrieve per claim.
    scrape : bool
        Whether to scrape full article text (slower but more accurate).
    """

    def __init__(
        self,
        pipeline_path: Optional[str | Path] = None,
        model_path: Optional[str | Path] = None,
        vectorizer_path: Optional[str | Path] = None,
        newsapi_key: Optional[str] = None,
        threshold: float = 0.45,
        w_ml: float = 0.40,
        w_sim: float = 0.40,
        w_cred: float = 0.20,
        max_sources: int = 5,
        scrape: bool = False,
    ):
        self.threshold = threshold
        self.w_ml = w_ml
        self.w_sim = w_sim
        self.w_cred = w_cred
        self.max_sources = max_sources
        self.scrape = scrape
        self.newsapi_key = newsapi_key or os.environ.get("NEWSAPI_KEY", "")

        # --- Load ML pipeline ---
        self._pipe = None
        self._clf = None
        self._vec = None
        self._load_model(pipeline_path, model_path, vectorizer_path)

        # --- Lazy imports of sub-modules ---
        from src.preprocessing.text_clean import clean_text
        from src.preprocessing.claim_extractor import extract_claims
        from src.retrieval.news_api import search_news
        from src.retrieval.web_scraper import extract_article_text
        from src.verification.similarity_checker import similarity_score, best_similarity
        from src.verification.credibility_scorer import credibility_score, average_credibility

        self._clean = clean_text
        self._extract_claims = extract_claims
        self._search = search_news
        self._scrape_url = extract_article_text
        self._sim = similarity_score
        self._best_sim = best_similarity
        self._cred = credibility_score
        self._avg_cred = average_credibility

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, pipeline_path, model_path, vectorizer_path):
        try:
            import joblib
        except ImportError:
            logger.error("joblib not installed. ML classifier disabled.")
            return

        if pipeline_path and Path(pipeline_path).exists():
            self._pipe = joblib.load(pipeline_path)
            logger.info("Loaded pipeline from %s", pipeline_path)
            return

        if model_path and vectorizer_path:
            mp, vp = Path(model_path), Path(vectorizer_path)
            if mp.exists() and vp.exists():
                self._clf = joblib.load(mp)
                self._vec = joblib.load(vp)
                logger.info("Loaded model + vectorizer.")
                return

        logger.warning(
            "No model artifacts found. ML score will default to 0.5 (neutral)."
        )

    def _ml_fake_prob(self, text: str) -> float:
        """Return P(FAKE) from ML model. Returns 0.5 if model not loaded."""
        if self._pipe is not None:
            return float(self._pipe.predict_proba([text])[0, 1])
        if self._clf is not None and self._vec is not None:
            X = self._vec.transform([text])
            return float(self._clf.predict_proba(X)[0, 1])
        return 0.5  # neutral

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, article_text: str, newsapi_key: Optional[str] = None) -> VerificationResult:
        """
        Run full verification pipeline on *article_text*.

        Parameters
        ----------
        article_text : str
            Raw headline or article body.
        newsapi_key : str, optional
            Override the key set at init time.

        Returns
        -------
        VerificationResult
        """
        api_key = newsapi_key or self.newsapi_key

        # 1. Clean
        clean = self._clean(article_text)

        # 2. Claims
        claims = self._extract_claims(article_text, max_claims=3)
        if not claims:
            claims = [clean[:200]]  # fallback: use first 200 chars

        # 3. Search
        all_articles: List[Dict] = []
        for claim in claims:
            hits = self._search(claim, api_key=api_key, max_results=self.max_sources)
            all_articles.extend(hits)

        # Deduplicate by URL
        seen_urls: set = set()
        unique_articles = []
        for a in all_articles:
            url = a.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(a)

        # 4. Scrape (optional)
        reference_texts: List[str] = []
        for a in unique_articles[: self.max_sources]:
            if self.scrape:
                full = self._scrape_url(a.get("url", ""))
                reference_texts.append(full or a.get("description", "") or a.get("title", ""))
            else:
                # Use title + description as lightweight reference
                snippet = (a.get("title") or "") + " " + (a.get("description") or "")
                reference_texts.append(snippet.strip())

        # 5. Semantic similarity
        if reference_texts:
            sim_scores = [self._sim(clean, r) for r in reference_texts if r]
            source_similarity = float(sum(sim_scores) / len(sim_scores)) if sim_scores else 0.0
        else:
            source_similarity = 0.0

        # 6. Source credibility
        source_urls = [a.get("url", a.get("source", "")) for a in unique_articles]
        source_credibility = self._avg_cred(source_urls) if source_urls else 0.35

        # 7. ML score  (1 - fake_prob = real probability)
        ml_fake_prob = self._ml_fake_prob(clean)
        ml_real_prob = 1.0 - ml_fake_prob

        # 8. Final credibility score
        # credibility = how "real" the article seems across all signals
        final_score = (
            self.w_ml   * ml_real_prob
            + self.w_sim  * source_similarity
            + self.w_cred * source_credibility
        )
        final_score = max(0.0, min(1.0, final_score))

        label = "REAL" if final_score >= self.threshold else "FAKE"

        # 9. Human-readable explanation
        lines = [
            f"ML model says {'REAL' if ml_real_prob > 0.5 else 'FAKE'} "
            f"(fake prob={ml_fake_prob:.0%})",
            f"Found {len(unique_articles)} source(s) via NewsAPI.",
            f"Average source similarity: {source_similarity:.0%}",
            f"Average source credibility: {source_credibility:.0%}",
            f"Combined credibility score: {final_score:.0%} → {label}",
        ]
        explanation = " | ".join(lines)

        return VerificationResult(
            label=label,
            credibility_score=final_score,
            ml_fake_probability=ml_fake_prob,
            source_similarity=source_similarity,
            source_credibility=source_credibility,
            supporting_sources=unique_articles[: self.max_sources],
            claims_extracted=claims,
            threshold=self.threshold,
            explanation=explanation,
        )
