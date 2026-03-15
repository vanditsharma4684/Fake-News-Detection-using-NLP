#!/usr/bin/env python3
"""
fake_news_classifier.py
-----------------------
Thin wrapper around the trained scikit-learn pipeline or
separate vectorizer + model artifacts.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["FakeNewsClassifier"]


class FakeNewsClassifier:
    """
    Wraps a trained TF-IDF + LogisticRegression (or RandomForest) pipeline.

    Parameters
    ----------
    pipeline_path : str | Path, optional
        Preferred: full sklearn Pipeline saved with joblib.
    model_path : str | Path, optional
        Classifier artifact (fallback).
    vectorizer_path : str | Path, optional
        TF-IDF vectorizer artifact (fallback).
    """

    def __init__(
        self,
        pipeline_path: Optional[str | Path] = None,
        model_path: Optional[str | Path] = None,
        vectorizer_path: Optional[str | Path] = None,
    ):
        import joblib

        self._pipe = None
        self._clf = None
        self._vec = None

        if pipeline_path and Path(pipeline_path).exists():
            self._pipe = joblib.load(pipeline_path)
            logger.info("Loaded pipeline from %s", pipeline_path)
        elif model_path and vectorizer_path:
            mp, vp = Path(model_path), Path(vectorizer_path)
            if mp.exists() and vp.exists():
                self._clf = joblib.load(mp)
                self._vec = joblib.load(vp)
                logger.info("Loaded model + vectorizer.")
            else:
                logger.warning("Model artifacts not found at given paths.")
        else:
            logger.warning("No model artifacts provided.")

    @property
    def loaded(self) -> bool:
        return self._pipe is not None or (self._clf is not None and self._vec is not None)

    def predict(self, text: str, threshold: float = 0.50) -> Tuple[str, float]:
        """
        Predict label and fake probability for *text*.

        Returns
        -------
        tuple (label, fake_probability)
            label is "FAKE" or "REAL".
        """
        prob = self.predict_proba(text)
        label = "FAKE" if prob >= threshold else "REAL"
        return label, prob

    def predict_proba(self, text: str) -> float:
        """Return P(FAKE) ∈ [0, 1]. Returns 0.5 if no model loaded."""
        if not self.loaded:
            return 0.5
        if self._pipe is not None:
            return float(self._pipe.predict_proba([text])[0, 1])
        X = self._vec.transform([text])
        return float(self._clf.predict_proba(X)[0, 1])
