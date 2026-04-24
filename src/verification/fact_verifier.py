#!/usr/bin/env python3
"""
fact_verifier.py
----------------
Orchestrates all components into a single verification pipeline.

Pipeline:
  1. Clean text
  2. Extract claims
  3. Search news sources for each claim (parallel)
  4. Optionally scrape full article text
  5. Compute semantic similarity between input and retrieved articles
  6. Score source credibility
  7. Run ML fake-news classifier
  8. Combine into a final credibility score [0, 1]

Fixes applied:
  Fix #3 — Zero-source dynamic weight fallback:
    When no external sources are found (API down, no key, obscure query),
    the old formula fed source_similarity=0.0 and source_credibility=0.35
    into the weighted sum, systematically penalising articles that simply
    couldn't be looked up. Now: zero sources → fall back to ML-only
    weighting (w_ml=1.0, w_sim=0.0, w_cred=0.0) so absence of external
    evidence is neutral, not punitive.

  Fix (executor timeout):
    ThreadPoolExecutor now passes timeout=20 to as_completed() so a
    hanging network request can't freeze the UI indefinitely.
"""
from __future__ import annotations
import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["FactVerifier", "VerificationResult"]


@dataclass
class VerificationResult:
    label: str                      # "REAL" or "FAKE"
    credibility_score: float        # 0.0 – 1.0  (higher = more credible)
    ml_fake_probability: float      # Raw ML model output
    source_similarity: float        # Avg similarity with retrieved articles
    source_credibility: float       # Avg credibility of retrieved sources
    supporting_sources: List[Dict]  # Articles found
    claims_extracted: List[str]     # Sentences sent to search
    threshold: float                # Decision threshold used
    sources_found: int = 0          # How many unique sources were retrieved
    explanation: str = ""           # Human-readable summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label":               self.label,
            "credibility_score":   round(self.credibility_score, 4),
            "ml_fake_probability": round(self.ml_fake_probability, 4),
            "source_similarity":   round(self.source_similarity, 4),
            "source_credibility":  round(self.source_credibility, 4),
            "supporting_sources":  self.supporting_sources,
            "claims_extracted":    self.claims_extracted,
            "threshold":           self.threshold,
            "sources_found":       self.sources_found,
            "explanation":         self.explanation,
        }


class FactVerifier:
    """
    Hybrid fake-news + fact-verification system.

    Parameters
    ----------
    pipeline_path : str | Path, optional
        Path to the trained scikit-learn pipeline (.joblib).
        Falls back to model_path + vectorizer_path if not given.
    model_path : str | Path, optional
    vectorizer_path : str | Path, optional
    newsapi_key : str, optional
        NewsAPI / NewsData key. Reads ``NEWSAPI_KEY`` env var if omitted.
    threshold : float
        Credibility score below this → FAKE. Default 0.45.
    w_ml : float
        Weight of ML classifier in final score (default 0.40).
    w_sim : float
        Weight of source similarity (default 0.40).
    w_cred : float
        Weight of source credibility (default 0.20).
    max_sources : int
        How many articles to retrieve per claim (default 5).
    scrape : bool
        Whether to scrape full article text (slower but more accurate).
    search_timeout : float
        Wall-clock seconds to wait for all claim searches before giving up
        (default 20). Prevents a single hanging request from freezing the UI.
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
        search_timeout: float = 20.0,
    ):
        self.threshold      = threshold
        self.w_ml           = w_ml
        self.w_sim          = w_sim
        self.w_cred         = w_cred
        self.max_sources    = max_sources
        self.scrape         = scrape
        self.search_timeout = search_timeout
        self.newsapi_key    = newsapi_key or os.environ.get("NEWSAPI_KEY", "")

        self._pipe = None
        self._clf  = None
        self._vec  = None
        self._load_model(pipeline_path, model_path, vectorizer_path)

        # Lazy imports — keep module-level startup fast
        from src.preprocessing.text_clean import clean_text
        from src.preprocessing.claim_extractor import extract_claims
        from src.retrieval.news_api import search_news
        from src.retrieval.web_scraper import extract_article_text
        from src.verification.similarity_checker import batch_similarity
        from src.verification.credibility_scorer import average_credibility
        from src.verification.contradiction_detector import batch_contradiction_detect

        self._clean        = clean_text
        self._extract      = extract_claims
        self._search       = search_news
        self._scrape_url   = extract_article_text
        self._batch_sim    = batch_similarity
        self._avg_cred     = average_credibility
        self._contradiction = batch_contradiction_detect

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

        logger.warning("No model artifacts found. ML score will default to 0.5 (neutral).")

    def _ml_fake_prob(self, text: str) -> float:
        """Return P(FAKE) from ML model. Returns 0.5 (neutral) if no model loaded."""
        if self._pipe is not None:
            return float(self._pipe.predict_proba([text])[0, 1])
        if self._clf is not None and self._vec is not None:
            X = self._vec.transform([text])
            return float(self._clf.predict_proba(X)[0, 1])
        return 0.5

    def _search_claims_parallel(
        self,
        claims: List[str],
        api_key: str,
        max_results: int,
        timeout: float,
    ) -> List[Dict]:
        """
        Search all claims in parallel, with a hard wall-clock timeout.
        Returns deduplicated article list.
        """
        all_articles: List[Dict] = []

        with ThreadPoolExecutor(max_workers=min(len(claims), 3)) as pool:
            futures = {
                pool.submit(self._search, claim, api_key=api_key, max_results=max_results): claim
                for claim in claims
            }
            try:
                for future in as_completed(futures, timeout=timeout):
                    try:
                        all_articles.extend(future.result())
                    except Exception as exc:
                        logger.warning("Claim search returned error: %s", exc)
            except _FuturesTimeout:
                logger.warning(
                    "Claim search timed out after %.0fs — proceeding with %d articles collected so far.",
                    timeout,
                    len(all_articles),
                )

        # Deduplicate by URL
        seen: set = set()
        unique: List[Dict] = []
        for a in all_articles:
            url = a.get("url", "")
            if url and url not in seen:
                seen.add(url)
                unique.append(a)

        return unique

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        article_text: str,
        newsapi_key: Optional[str] = None,
        *,
        threshold: Optional[float] = None,
        w_ml: Optional[float] = None,
        w_sim: Optional[float] = None,
        w_cred: Optional[float] = None,
        max_sources: Optional[int] = None,
        scrape: Optional[bool] = None,
        search_timeout: Optional[float] = None,
    ) -> VerificationResult:
        """
        Run full verification pipeline on *article_text*.

        All tunable parameters can be overridden per-call without mutating
        instance state, making this method safe for concurrent use.

        Parameters
        ----------
        article_text : str
            Raw headline or article body.
        newsapi_key : str, optional
            Override the key set at init time.
        threshold : float, optional   Override credibility threshold.
        w_ml : float, optional        Override ML weight.
        w_sim : float, optional       Override similarity weight.
        w_cred : float, optional      Override credibility weight.
        max_sources : int, optional   Override max sources per claim.
        scrape : bool, optional       Override scrape toggle.
        search_timeout : float, optional  Override search wall-clock timeout.

        Returns
        -------
        VerificationResult
        """
        # Per-call parameters — never mutates self
        _threshold      = threshold      if threshold      is not None else self.threshold
        _w_ml           = w_ml           if w_ml           is not None else self.w_ml
        _w_sim          = w_sim          if w_sim          is not None else self.w_sim
        _w_cred         = w_cred         if w_cred         is not None else self.w_cred
        _max_sources    = max_sources    if max_sources     is not None else self.max_sources
        _scrape         = scrape         if scrape          is not None else self.scrape
        _search_timeout = search_timeout if search_timeout  is not None else self.search_timeout

        api_key = newsapi_key or self.newsapi_key

        # ── 1. Clean ─────────────────────────────────────────────────────
        clean = self._clean(article_text)

        # ── 2. Extract claims ─────────────────────────────────────────────
        claims = self._extract(article_text, max_claims=3)
        if not claims:
            claims = [clean[:200]]

        # ── 3. Search (parallel, with timeout) ───────────────────────────
        unique_articles = self._search_claims_parallel(
            claims, api_key, _max_sources, _search_timeout
        )

        # ── 4. Build reference texts ──────────────────────────────────────
        reference_texts: List[str] = []
        for a in unique_articles[:_max_sources]:
            if _scrape:
                full = self._scrape_url(a.get("url", ""))
                reference_texts.append(
                    full or a.get("description", "") or a.get("title", "")
                )
            else:
                snippet = (a.get("title") or "") + " " + (a.get("description") or "")
                reference_texts.append(snippet.strip())

        # ── 5. Semantic similarity + Contradiction penalty ────────────────
        contradiction_penalty = 0.0
        source_similarity     = 0.0

        if reference_texts:
            sim_scores = self._batch_sim(clean, reference_texts)

            # Contradiction check only on high-similarity references
            high_sim_refs = [
                ref for score, ref in zip(sim_scores, reference_texts)
                if score > 0.40
            ]
            if high_sim_refs:
                c_scores = self._contradiction(clean, high_sim_refs)
                if c_scores:
                    # Cap penalty at 0.25 — a contradiction signal should
                    # influence, not single-handedly determine, the verdict
                    contradiction_penalty = min(max(c_scores) * 0.35, 0.25)

            non_zero = [s for s in sim_scores if s > 0.0]
            if non_zero:
                # Top-3 average: resistant to dilution by irrelevant results
                top_k = sorted(non_zero, reverse=True)[:3]
                source_similarity = float(sum(top_k) / len(top_k))

        # ── 6. Source credibility ─────────────────────────────────────────
        # Prefer 'credibility_url' (real publisher domain) over 'url' —
        # Google News RSS articles have redirect URLs (news.google.com/...)
        # in 'url' but the actual publisher URL in 'credibility_url'.
        # Fall back to 'source' (NewsData short ID) if no URL is available.
        cred_inputs = [
            a.get("credibility_url") or a.get("url") or a.get("source", "")
            for a in unique_articles
        ]
        source_credibility = self._avg_cred(cred_inputs) if cred_inputs else 0.35

        # ── 7. ML classification ──────────────────────────────────────────
        ml_fake_prob = self._ml_fake_prob(clean)
        ml_real_prob = 1.0 - ml_fake_prob

        # ── 8. Score fusion ───────────────────────────────────────────────
        #
        # FIX #3: Dynamic weight fallback when no external sources found.
        #
        # Problem: With zero sources, source_similarity=0.0 and
        # source_credibility=0.35 enter the formula, systematically
        # penalising articles that just happen to be unsearchable.
        # Absence of external evidence ≠ evidence of fakeness.
        #
        # Solution: When no sources are found, fall back to ML-only
        # weighting. The similarity and credibility components cannot
        # contribute useful signal if there are no sources to score.
        #
        if not unique_articles:
            # No external evidence — use ML signal only
            w_ml_eff, w_sim_eff, w_cred_eff = 1.0, 0.0, 0.0
            logger.info(
                "No external sources found — using ML-only weighting "
                "(w_ml=1.0, w_sim=0.0, w_cred=0.0)."
            )
        else:
            # Normalise provided weights so they always sum to 1.0
            total = _w_ml + _w_sim + _w_cred
            if total > 0:
                w_ml_eff   = _w_ml   / total
                w_sim_eff  = _w_sim  / total
                w_cred_eff = _w_cred / total
            else:
                w_ml_eff, w_sim_eff, w_cred_eff = 1.0, 0.0, 0.0

        raw_score = (
            w_ml_eff   * ml_real_prob
            + w_sim_eff  * source_similarity
            + w_cred_eff * source_credibility
        )

        final_score = max(0.0, min(1.0, raw_score - contradiction_penalty))
        label = "REAL" if final_score >= _threshold else "FAKE"

        # ── 9. Human-readable explanation ─────────────────────────────────
        lines = [
            f"ML model says {'REAL' if ml_real_prob > 0.5 else 'FAKE'} "
            f"(fake prob={ml_fake_prob:.0%})",
        ]

        if not unique_articles:
            lines.append(
                "No external sources found — verdict based on ML signal only. "
                "This does NOT count against the article's credibility."
            )
        else:
            lines.append(f"Found {len(unique_articles)} source(s) via search.")
            lines.append(f"Source similarity (top-3 avg): {source_similarity:.0%}")
            lines.append(f"Source credibility (avg): {source_credibility:.0%}")

        if contradiction_penalty > 0:
            lines.append(
                f"Contradiction signal detected in similar sources "
                f"(penalty: -{contradiction_penalty:.0%})"
            )

        lines.append(
            f"Weights used — ML:{w_ml_eff:.0%} | "
            f"Similarity:{w_sim_eff:.0%} | "
            f"Credibility:{w_cred_eff:.0%}"
        )
        lines.append(f"Final credibility score: {final_score:.0%} → {label}")

        return VerificationResult(
            label=label,
            credibility_score=final_score,
            ml_fake_probability=ml_fake_prob,
            source_similarity=source_similarity,
            source_credibility=source_credibility,
            supporting_sources=unique_articles[:_max_sources],
            claims_extracted=claims,
            threshold=_threshold,
            sources_found=len(unique_articles),
            explanation=" | ".join(lines),
        )
