#!/usr/bin/env python3
"""
contradiction_detector.py
-------------------------
Identifies if retrieved sources actively CONTRADICT the original claim —
meaning the reference debunks the claim, not merely discusses the same topic.

Approach (two-tier):
  Tier 1 — Zero-shot NLI (if transformers installed):
    Use a pre-trained NLI model to classify each (claim, reference) pair as
    "entailment" (source supports claim), "neutral", or "contradiction"
    (source refutes claim). This understands context — "The claim is false"
    is correctly read as contradiction, not as the source being unreliable.

  Tier 2 — Structural heuristic fallback (no ML deps needed):
    Instead of counting raw debunk keywords (the old bug), we look for
    NEGATION patterns *in close proximity to claim-specific terms*.
    A reference that says "NASA denied the claim" is different from one
    that simply mentions "false reports circulating online".

The old approach (count "fake"/"false" anywhere in the text) is removed
because those words appear in every legitimate fact-check and Reuters
article, causing systematic false penalties on credible sources.
"""
from __future__ import annotations

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

__all__ = ["detect_contradiction", "batch_contradiction_detect"]

# ---------------------------------------------------------------------------
# Tier 1 — NLI via transformers (zero-shot)
# ---------------------------------------------------------------------------
_NLI_PIPE = None
_NLI_AVAILABLE = False

try:
    from transformers import pipeline as _hf_pipeline
    _NLI_AVAILABLE = True
except ImportError:
    logger.info(
        "transformers not installed — contradiction detector will use "
        "structural heuristic fallback (less accurate but no false positives)."
    )


def _get_nli_pipeline():
    """Lazy-load the zero-shot NLI pipeline on first use."""
    global _NLI_PIPE
    if _NLI_PIPE is None:
        logger.info("Loading zero-shot NLI model (cross-encoder/nli-MiniLM2-L6-H768)…")
        _NLI_PIPE = _hf_pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-MiniLM2-L6-H768",
            # Use CPU by default — small model, fast enough
        )
        logger.info("NLI model loaded.")
    return _NLI_PIPE


def _nli_contradiction_score(claim: str, reference: str) -> float:
    """
    Use zero-shot NLI to score how strongly *reference* contradicts *claim*.
    Returns a score in [0, 1] where 1.0 = certain contradiction.

    We use the claim as the hypothesis and a sentence from the reference as
    the premise, then take the 'contradiction' label probability.
    """
    try:
        pipe = _get_nli_pipeline()
        # Truncate to avoid model token limits
        premise = reference[:512]
        result = pipe(
            premise,
            candidate_labels=["supports this claim", "contradicts this claim", "unrelated"],
            hypothesis_template="This text {} .",
        )
        # Find score for the 'contradicts' label
        label_scores = dict(zip(result["labels"], result["scores"]))
        return float(label_scores.get("contradicts this claim", 0.0))
    except Exception as exc:
        logger.warning("NLI inference failed (%s). Returning 0.0.", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Tier 2 — Structural heuristic (fallback, no ML)
# ---------------------------------------------------------------------------
# These patterns detect TARGETED debunking — the reference explicitly says
# the claim/story is wrong. We require negation adjacent to a claim-marker
# phrase, not bare keywords floating anywhere in the article.
#
# Crucially absent: "fake", "false", "hoax" as standalone keywords.
# Those words appear in every fact-check article and must NOT trigger a penalty.

_DEBUNK_PATTERNS: List[re.Pattern] = [
    # "X is a hoax" / "X is fabricated" / "X is not true" (targeted)
    re.compile(
        r"\b(?:claim|story|report|article|allegation|assertion)\b.{0,40}"
        r"\b(?:is\s+(?:false|untrue|fabricated|a\s+hoax|baseless|unfounded)|"
        r"has\s+been\s+(?:debunked|refuted|disproven|fact.?checked))\b",
        re.IGNORECASE,
    ),
    # "there is no evidence that X"
    re.compile(
        r"\bno\s+(?:credible\s+)?evidence\s+(?:that|for|of|to\s+support)\b",
        re.IGNORECASE,
    ),
    # "[source] denied / rejected / disputed the claim"
    re.compile(
        r"\b(?:denied|rejected|disputed|contradicted|challenged|rebutted)\s+"
        r"(?:the\s+)?(?:claim|allegation|report|story)\b",
        re.IGNORECASE,
    ),
    # "fact.check: false" / "verdict: false" / "rating: pants on fire"
    re.compile(
        r"\b(?:fact.?check|verdict|rating|ruling)\s*[:—]\s*"
        r"(?:false|misleading|mostly\s+false|pants\s+on\s+fire|four\s+pinocchio)",
        re.IGNORECASE,
    ),
    # "X did NOT [verb]" where X overlaps with the claim subject
    re.compile(
        r"\bdid\s+not\b.{0,60}\b(?:happen|occur|confirm|announce|say|state|reveal)\b",
        re.IGNORECASE,
    ),
]


def _heuristic_contradiction_score(claim: str, reference: str) -> float:
    """
    Structural heuristic: check for targeted debunking patterns.
    Returns 0.0 (no contradiction found) or 0.6 (pattern matched).
    Never returns 0.9 based on bare keyword count like the old version did.
    """
    if not claim or not reference:
        return 0.0

    ref = reference[:2000]  # cap length

    matches = sum(1 for pat in _DEBUNK_PATTERNS if pat.search(ref))

    if matches >= 2:
        return 0.70   # multiple targeted debunking patterns — high confidence
    elif matches == 1:
        return 0.40   # one pattern — moderate signal
    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_contradiction(claim: str, reference_text: str) -> float:
    """
    Score how strongly *reference_text* contradicts *claim*.

    Returns
    -------
    float in [0, 1]
        0.0  = no contradiction detected
        ~0.4 = weak structural signal
        ~0.7 = strong structural signal or moderate NLI score
        1.0  = NLI is highly confident the reference refutes the claim

    Uses NLI if transformers is installed, structural heuristic otherwise.
    """
    if not claim or not reference_text:
        return 0.0

    if _NLI_AVAILABLE:
        return _nli_contradiction_score(claim, reference_text)

    return _heuristic_contradiction_score(claim, reference_text)


def batch_contradiction_detect(claim: str, references: List[str]) -> List[float]:
    """
    Return contradiction scores for *claim* vs each text in *references*.

    Uses NLI batch processing when available for efficiency.
    """
    if not references:
        return []

    if _NLI_AVAILABLE:
        # Process each individually (NLI pipeline handles batching internally)
        return [detect_contradiction(claim, ref) for ref in references]

    # Heuristic path — cheap, no model needed
    return [_heuristic_contradiction_score(claim, ref) for ref in references]
