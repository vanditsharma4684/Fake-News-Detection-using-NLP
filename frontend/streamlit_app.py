#!/usr/bin/env python3
"""
streamlit_app.py
----------------
Upgraded Fake News & Misinformation Verifier — Streamlit UI

Features:
  • ML-only quick prediction (no API key needed)
  • Full multi-source verification (requires NewsAPI key)
  • Source credibility breakdown
  • Semantic similarity display
  • Supporting sources list with links

Run:
    streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations
import sys
import os
import re
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Verifier v2",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.main-header {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #e63946, #457b9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sub-header {
    color: #6c757d;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}
.verdict-real {
    background: #d4edda;
    border-left: 6px solid #28a745;
    padding: 1rem 1.5rem;
    border-radius: 8px;
    font-size: 1.5rem;
    font-weight: 700;
    color: #155724;
}
.verdict-fake {
    background: #f8d7da;
    border-left: 6px solid #dc3545;
    padding: 1rem 1.5rem;
    border-radius: 8px;
    font-size: 1.5rem;
    font-weight: 700;
    color: #721c24;
}
.source-card {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.metric-box {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.claim-chip {
    display: inline-block;
    background: #e9ecef;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    margin: 0.2rem;
    font-size: 0.85rem;
    color: #343a40;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("### 🗂️ Model Artifacts")
    models_dir = ROOT / "models"
    pipeline_path   = models_dir / "pipeline.joblib"
    model_path      = models_dir / "fake_news_model.joblib"
    vectorizer_path = models_dir / "vectorizer.joblib"

    model_ok = pipeline_path.exists() or (model_path.exists() and vectorizer_path.exists())
    st.markdown(
        f"{'✅' if model_ok else '❌'} ML Model: "
        f"{'Loaded' if model_ok else 'Not found — run train_model.py first'}"
    )

    st.markdown("---")
    st.markdown("### 🔑 News API Key")

    st.markdown(
        """
**Supported providers — paste the key only, not the full URL:**

| Provider | Key format | Free tier |
|----------|-----------|-----------|
| [NewsData.io](https://newsdata.io/) | `pub_xxxx...` | 200 req/day ✅ |
| [NewsAPI.org](https://newsapi.org/) | 32 hex chars | Dev only ✅ |
        """,
    )

    raw_key_input = st.text_input(
        "Paste your API key here",
        type="password",
        placeholder="pub_xxxxx...  OR  abc123def456...",
        help="NewsData key starts with pub_  |  NewsAPI key is 32 hex characters",
    )

    # Clean and validate the pasted key
    newsapi_key = ""
    if raw_key_input:
        import re as _re
        k = raw_key_input.strip().strip('"').strip("'")
        # If they accidentally pasted a full URL, extract the key param
        m = _re.search(r"apikey=([^\s&]+)", k, _re.IGNORECASE)
        if m:
            k = m.group(1)
            st.warning("⚠️ Looks like you pasted a full URL. Extracted key: `" + k[:8] + "…`")
        # Detect provider
        if k.startswith("pub_"):
            newsapi_key = k
            st.success(f"✅ NewsData.io key detected (`pub_...{k[-4:]}`)")
        elif len(k) == 32 and _re.fullmatch(r"[a-f0-9]{32}", k):
            newsapi_key = k
            st.success(f"✅ NewsAPI.org key detected (`...{k[-4:]}`)")
        elif len(k) > 8:
            # Accept it anyway but warn
            newsapi_key = k
            st.info(f"ℹ️ Key accepted — provider will be auto-detected.")
        else:
            st.error("❌ Key too short. Please paste the full API key.")

    if not newsapi_key:
        # Try environment variables
        for env_var in ("NEWSAPI_KEY", "NEWSDATA_KEY"):
            env_key = os.environ.get(env_var, "")
            if env_key:
                newsapi_key = env_key.strip()
                st.success(f"Using `{env_var}` from environment.")
                break

    st.markdown("---")
    st.markdown("### 🎚️ Settings")
    threshold = st.slider(
        "Credibility threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Score ≥ threshold → REAL, else → FAKE",
    )
    scrape_full = st.checkbox(
        "Scrape full article text",
        value=False,
        help="Slower but more accurate similarity. Requires newspaper3k.",
    )
    max_sources = st.slider("Max sources to retrieve", 2, 10, 5)

    st.markdown("---")
    st.markdown("### 📐 Score Weights")
    w_ml   = st.slider("ML classifier weight",       0.0, 1.0, 0.40, 0.05)
    w_sim  = st.slider("Source similarity weight",   0.0, 1.0, 0.40, 0.05)
    w_cred = st.slider("Source credibility weight",  0.0, 1.0, 0.20, 0.05)

    total_w = w_ml + w_sim + w_cred
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"Weights sum to {total_w:.2f} (ideally 1.0). They'll be normalised automatically.")

    st.markdown("---")
    st.markdown(
        "<small>Fake News Verifier v2.0 — NLP + Multi-Source Verification</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Normalise weights
# ---------------------------------------------------------------------------
total_w = w_ml + w_sim + w_cred
if total_w > 0:
    w_ml, w_sim, w_cred = w_ml / total_w, w_sim / total_w, w_cred / total_w

# ---------------------------------------------------------------------------
# Load verifier and classifier (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ML model…")
def load_verifier(w_ml, w_sim, w_cred, threshold, max_sources):
    from src.verification.fact_verifier import FactVerifier
    return FactVerifier(
        pipeline_path=pipeline_path if pipeline_path.exists() else None,
        model_path=model_path if model_path.exists() else None,
        vectorizer_path=vectorizer_path if vectorizer_path.exists() else None,
        threshold=threshold,
        w_ml=w_ml,
        w_sim=w_sim,
        w_cred=w_cred,
        max_sources=max_sources,
        scrape=False,  # controlled at runtime
    )

@st.cache_resource(show_spinner=False)
def load_classifier():
    from src.models.fake_news_classifier import FakeNewsClassifier
    return FakeNewsClassifier(
        pipeline_path=pipeline_path if pipeline_path.exists() else None,
        model_path=model_path if model_path.exists() else None,
        vectorizer_path=vectorizer_path if vectorizer_path.exists() else None,
    )

verifier   = load_verifier(w_ml, w_sim, w_cred, threshold, max_sources)
classifier = load_classifier()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🔍 Fake News & Misinformation Verifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">ML classifier + multi-source fact verification using NewsAPI & semantic similarity</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
col_input, col_tip = st.columns([3, 1])

with col_input:
    article_text = st.text_area(
        "Paste your news headline or article:",
        height=180,
        placeholder="e.g. Scientists discover new planet capable of supporting life…",
    )

with col_tip:
    st.markdown("##### 💡 Tips")
    st.markdown(
        "- Paste full paragraph for best results\n"
        "- Short headlines may have lower confidence\n"
        "- Add a NewsAPI key for source verification\n"
        "- Adjust weights in the sidebar"
    )

col_btn1, col_btn2, col_spacer = st.columns([1, 1.5, 4])

with col_btn1:
    run_ml = st.button("⚡ Quick Predict", use_container_width=True, help="ML only — instant")

with col_btn2:
    run_full = st.button(
        "🌐 Verify with Sources",
        use_container_width=True,
        type="primary",
        help="Full pipeline with NewsAPI source search",
        disabled=not bool(newsapi_key),
    )
    if not newsapi_key:
        st.caption("⚠️ News API key required (see sidebar)")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def score_bar(score: float, label: str) -> None:
    colour = "#28a745" if label == "REAL" else "#dc3545"
    pct = int(score * 100)
    st.markdown(
        f"""
        <div style="margin:0.5rem 0">
            <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:4px">
                <span>Credibility Score</span><span><b>{pct}%</b></span>
            </div>
            <div style="background:#e9ecef;border-radius:6px;height:14px">
                <div style="width:{pct}%;background:{colour};height:14px;border-radius:6px;
                            transition:width 0.4s ease"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mini_bar(value: float, label: str, colour: str = "#457b9d") -> None:
    pct = int(value * 100)
    st.markdown(
        f"""
        <div style="margin:0.3rem 0">
            <div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:2px">
                <span>{label}</span><span>{pct}%</span>
            </div>
            <div style="background:#e9ecef;border-radius:4px;height:8px">
                <div style="width:{pct}%;background:{colour};height:8px;border-radius:4px"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_sources(sources: list) -> None:
    if not sources:
        st.info("No external sources retrieved.")
        return
    for s in sources:
        title  = s.get("title") or "Untitled"
        url    = s.get("url") or "#"
        src    = s.get("source") or "Unknown"
        desc   = s.get("description") or ""
        pub    = s.get("publishedAt", "")[:10]

        from src.verification.credibility_scorer import credibility_score
        cred = credibility_score(url)
        cred_icon = "🟢" if cred >= 0.85 else "🟡" if cred >= 0.65 else "🔴"

        st.markdown(
            f"""
            <div class="source-card">
                <b>{cred_icon} <a href="{url}" target="_blank">{title}</a></b><br>
                <small>📰 {src} &nbsp;|&nbsp; 📅 {pub} &nbsp;|&nbsp; 
                Credibility: <b>{cred:.0%}</b></small>
                {"<br><small style='color:#6c757d'>" + desc[:200] + "…</small>" if desc else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Quick Predict (ML only)
# ---------------------------------------------------------------------------
if run_ml and article_text.strip():
    st.markdown("---")
    st.markdown("### ⚡ Quick ML Prediction")

    from src.preprocessing.text_clean import clean_text
    cleaned = clean_text(article_text)

    if not classifier.loaded:
        st.error("ML model not loaded. Run `train_model.py` first.")
    else:
        label, prob_fake = classifier.predict(cleaned, threshold=0.50)
        prob_real = 1.0 - prob_fake

        verdict_class = "verdict-real" if label == "REAL" else "verdict-fake"
        verdict_icon  = "✅" if label == "REAL" else "🚫"

        st.markdown(
            f'<div class="{verdict_class}">{verdict_icon} Prediction: {label}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        c1, c2, c3 = st.columns(3)
        c1.metric("Label",            label)
        c2.metric("Fake Probability", f"{prob_fake:.1%}")
        c3.metric("Real Probability", f"{prob_real:.1%}")

        st.markdown("")
        mini_bar(prob_real, "Real probability",  "#28a745")
        mini_bar(prob_fake, "Fake probability",  "#dc3545")

        st.caption("_ML-only prediction. Add a NewsAPI key and click 'Verify with Sources' for full analysis._")

elif run_ml and not article_text.strip():
    st.warning("Please paste some text first.")

# ---------------------------------------------------------------------------
# Full Verification
# ---------------------------------------------------------------------------
if run_full and article_text.strip():
    st.markdown("---")
    st.markdown("### 🌐 Full Multi-Source Verification")

    with st.spinner("Extracting claims, searching sources, computing similarity…"):
        try:
            verifier.threshold = threshold
            verifier.w_ml      = w_ml
            verifier.w_sim     = w_sim
            verifier.w_cred    = w_cred
            verifier.max_sources = max_sources
            verifier.scrape    = scrape_full

            result = verifier.verify(article_text, newsapi_key=newsapi_key)

        except Exception as exc:
            st.error(f"Verification failed: {exc}")
            st.stop()

    # ----- Verdict -----
    verdict_class = "verdict-real" if result.label == "REAL" else "verdict-fake"
    verdict_icon  = "✅" if result.label == "REAL" else "🚫"
    st.markdown(
        f'<div class="{verdict_class}">{verdict_icon} Verdict: {result.label} '
        f'— Credibility Score: {result.credibility_score:.0%}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ----- Score bar -----
    score_bar(result.credibility_score, result.label)
    st.markdown("")

    # ----- Metrics grid -----
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Score",         f"{result.credibility_score:.0%}")
    m2.metric("ML Fake Probability",   f"{result.ml_fake_probability:.0%}")
    m3.metric("Source Similarity",     f"{result.source_similarity:.0%}")
    m4.metric("Source Credibility",    f"{result.source_credibility:.0%}")

    st.markdown("")

    # ----- Component breakdown -----
    with st.expander("📊 Score Breakdown", expanded=True):
        mini_bar(1 - result.ml_fake_probability, "ML real probability (higher = more real)", "#28a745")
        mini_bar(result.source_similarity,       "Source semantic similarity",               "#457b9d")
        mini_bar(result.source_credibility,      "Source credibility average",               "#f4a261")
        st.caption(
            f"Weights used → ML: {w_ml:.0%}, Similarity: {w_sim:.0%}, Credibility: {w_cred:.0%} | "
            f"Threshold: {threshold:.0%}"
        )

    # ----- Claims extracted -----
    with st.expander("🔎 Claims Sent to Search", expanded=False):
        if result.claims_extracted:
            for i, c in enumerate(result.claims_extracted, 1):
                st.markdown(f"**{i}.** {c}")
        else:
            st.info("No claims extracted.")

    # ----- Supporting sources -----
    with st.expander(f"📰 Retrieved Sources ({len(result.supporting_sources)})", expanded=True):
        show_sources(result.supporting_sources)

    # ----- Explanation -----
    with st.expander("💬 Explanation"):
        for line in result.explanation.split(" | "):
            st.markdown(f"- {line}")

elif run_full and not article_text.strip():
    st.warning("Please paste some text first.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<small>Built with scikit-learn · sentence-transformers · NewsAPI · Streamlit &nbsp;|&nbsp; "
    "For research & educational use only.</small>",
    unsafe_allow_html=True,
)