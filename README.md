# 🔍 Fake News & Misinformation Verifier v2.0

> **Upgraded from** a pattern-based ML classifier  
> **To** a full hybrid AI fact-verification system

---

## Table of Contents
- [Overview](#overview)
- [What Changed](#what-changed)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [How It Works](#how-it-works)
- [Technologies](#technologies)
- [Future Improvements](#future-improvements)

---

## Overview

Version 2 extends the original TF-IDF + Logistic Regression classifier with a **multi-source fact-verification pipeline**. The system now answers not just *"does this look like fake news stylistically?"* but also *"is this claim supported by credible, external sources?"*

---

## What Changed

| Feature | v1 (Original) | v2 (Upgraded) |
|---------|--------------|---------------|
| ML Classifier | ✅ TF-IDF + LogReg | ✅ Same (retained) |
| Claim extraction | ❌ | ✅ spaCy sentence splitter |
| News search | ❌ | ✅ NewsAPI integration |
| Web scraping | ❌ | ✅ newspaper3k + BS4 |
| Semantic similarity | ❌ | ✅ Sentence-BERT / TF-IDF |
| Source credibility | ❌ | ✅ 40+ trusted domains rated |
| Combined score | ❌ | ✅ Weighted fusion |
| REST API | ❌ | ✅ FastAPI (`/verify`, `/predict`) |
| UI | ✅ Basic Streamlit | ✅ Full dashboard with source cards |

---

## Architecture

```
User Input (news article)
        │
        ▼
  Text Cleaning  (src/preprocessing/text_clean.py)
        │
        ▼
  Claim Extraction  (src/preprocessing/claim_extractor.py)
        │
        ▼
  NewsAPI Search  (src/retrieval/news_api.py)
        │
        ▼
  [Optional] Web Scraping  (src/retrieval/web_scraper.py)
        │
        ▼
  Semantic Similarity  (src/verification/similarity_checker.py)
  [Sentence-BERT or TF-IDF fallback]
        │
        ▼
  Source Credibility Scoring  (src/verification/credibility_scorer.py)
        │
        ▼
  ML Fake News Classifier  (src/models/fake_news_classifier.py)
        │
        ▼
  Score Fusion: 0.4×ML + 0.4×Similarity + 0.2×Credibility
        │
        ▼
  Final Verdict + Evidence Report
```

---

## Project Structure

```
fake-news-verifier/
│
├── data/
│   ├── raw/
│   │   ├── True.csv              ← original real news dataset
│   │   └── Fake.csv              ← original fake news dataset
│   └── processed/
│
├── models/                       ← saved after running train_model.py
│   ├── pipeline.joblib
│   ├── vectorizer.joblib
│   ├── fake_news_model.joblib
│   └── charts/
│
├── src/
│   ├── preprocessing/
│   │   ├── text_clean.py         ← URL/email/whitespace cleaning
│   │   └── claim_extractor.py    ← spaCy sentence-level claim extraction
│   │
│   ├── retrieval/
│   │   ├── news_api.py           ← NewsAPI search wrapper
│   │   └── web_scraper.py        ← newspaper3k + BS4 full-text scraper
│   │
│   ├── verification/
│   │   ├── similarity_checker.py ← Sentence-BERT / TF-IDF cosine similarity
│   │   ├── credibility_scorer.py ← 40+ domain credibility database
│   │   └── fact_verifier.py      ← main orchestration engine
│   │
│   ├── models/
│   │   └── fake_news_classifier.py ← joblib model wrapper
│   │
│   ├── api/
│   │   └── server.py             ← FastAPI REST backend
│   │
│   └── utils/
│       └── helpers.py            ← JSON I/O, directory helpers
│
├── frontend/
│   └── streamlit_app.py          ← upgraded Streamlit dashboard
│
├── outputs/
│   ├── charts/
│   └── logs/
│
├── train_model.py                ← training entry point
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone / navigate to project
cd fake-news-verifier

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install spaCy model (for claim extraction)
python -m spacy download en_core_web_sm
```

---

## Setup

### Step 1 — Train the ML model

```bash
python train_model.py \
  --real data/raw/True.csv \
  --fake data/raw/Fake.csv \
  --outdir models
```

This saves `pipeline.joblib`, `vectorizer.joblib`, `fake_news_model.joblib` and evaluation charts to `models/`.

### Step 2 — Get a NewsAPI key (free)

1. Visit [https://newsapi.org/](https://newsapi.org/)
2. Register for a free account
3. Copy your API key
4. Either paste it in the Streamlit sidebar, or set it as an environment variable:

```bash
# Linux / macOS
export NEWSAPI_KEY=your_key_here

# Windows
set NEWSAPI_KEY=your_key_here
```

---

## Usage

### Streamlit Web App

```bash
streamlit run frontend/streamlit_app.py
```

Open `http://localhost:8501`

**UI features:**
- **⚡ Quick Predict** — ML only, instant, no API key needed
- **🌐 Verify with Sources** — full pipeline with NewsAPI (key required)
- Adjustable weights and threshold in sidebar
- Source cards with credibility ratings
- Score breakdown chart

### FastAPI REST Backend

```bash
uvicorn src.api.server:app --reload --port 8000
```

API docs at `http://localhost:8000/docs`

---

## API Reference

### `GET /health`
```json
{ "status": "ok", "ml_model_loaded": true }
```

### `POST /predict` — ML only
```json
{
  "text": "NASA confirms asteroid will hit Earth.",
  "threshold": 0.50
}
```
Response:
```json
{
  "label": "FAKE",
  "fake_probability": 0.72,
  "threshold": 0.5
}
```

### `POST /verify` — Full pipeline
```json
{
  "text": "NASA confirms asteroid will hit Earth.",
  "newsapi_key": "YOUR_KEY",
  "threshold": 0.45
}
```
Response:
```json
{
  "label": "FAKE",
  "credibility_score": 0.18,
  "ml_fake_probability": 0.72,
  "source_similarity": 0.06,
  "source_credibility": 0.35,
  "supporting_sources": [...],
  "claims_extracted": [...],
  "explanation": "..."
}
```

---

## How It Works

### Score Formula
```
credibility = w_ml × (1 - fake_prob)
            + w_sim × source_similarity
            + w_cred × source_credibility

Default weights: w_ml=0.40, w_sim=0.40, w_cred=0.20

label = "REAL" if credibility ≥ threshold else "FAKE"
```

### Source Credibility Tiers

| Tier | Score | Examples |
|------|-------|---------|
| Wire services | 0.95–0.97 | Reuters, AP, AFP |
| Public broadcasters | 0.90–0.95 | BBC, NPR, PBS |
| Major outlets | 0.80–0.90 | NYT, Guardian, CNN |
| General outlets | 0.65–0.79 | Various national media |
| Unknown | 0.35 | Any unrecognised domain |

### Semantic Similarity
- **Sentence-BERT** (`all-MiniLM-L6-v2`) if `sentence-transformers` installed
- Falls back to **TF-IDF cosine similarity** automatically

---

## Technologies

| Component | Tool |
|-----------|------|
| ML classifier | scikit-learn (TF-IDF + Logistic Regression) |
| Claim extraction | spaCy |
| Semantic similarity | sentence-transformers (SBERT) |
| News retrieval | NewsAPI |
| Web scraping | newspaper3k, BeautifulSoup4 |
| REST API | FastAPI + uvicorn |
| Web UI | Streamlit |
| Serialisation | joblib |

---

## Future Improvements

- **BERT / DistilBERT** for deeper contextual classification
- **Knowledge graph** verification (Wikidata / DBpedia)
- **Temporal consistency** — flag claims that contradict historical articles
- **Multi-language** support
- **LIME / SHAP** explainability for model decisions
- **Caching layer** (Redis) to avoid repeat API calls
- **Deploy** on Streamlit Cloud or Hugging Face Spaces
