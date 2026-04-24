# 🛠️ Fake News Verifier — Improvement Plan

I audited every file in the codebase. Below are **real bugs, performance bottlenecks, accuracy gaps, and feature improvements** — ranked by impact.

---

## 🔴 Critical Bugs (Fix Immediately)

### 1. Race Condition — FastAPI `/verify` Mutates Shared Singleton

**File:** [server.py, lines 118-119](file:///c:/Users/VANDIT%20SHARMA/OneDrive/Desktop/CollegeProject/vandit/fake-news-verifier/src/api/server.py#L118-L119)

The `/verify` endpoint **mutates the global `_verifier` object** before each request:
```python
_verifier.threshold = req.threshold   # ← shared across ALL requests
result = _verifier.verify(req.text, ...)
```
If two users send concurrent requests with different thresholds, **one overwrites the other's settings** mid-processing. Same issue exists in `streamlit_app.py` lines 405-410.

**Fix:** Pass parameters to `.verify()` instead of mutating instance state, or create a new `FactVerifier` per request.

```diff
- _verifier.threshold = req.threshold
- result = _verifier.verify(req.text, newsapi_key=req.newsapi_key)
+ result = _verifier.verify(
+     req.text,
+     newsapi_key=req.newsapi_key,
+     threshold=req.threshold,   # pass as argument
+ )
```

---

### 2. Credibility Scorer — False Positive Domain Matching

**File:** [credibility_scorer.py, line 117](file:///c:/Users/VANDIT%20SHARMA/OneDrive/Desktop/CollegeProject/vandit/fake-news-verifier/src/verification/credibility_scorer.py#L116-L118)

```python
if domain.endswith(known_domain) or known_domain.endswith(domain):
    return score
```

The **reverse check** `known_domain.endswith(domain)` is dangerously loose:
- `domain = "uk"` → matches `"bbc.co.uk"` (score 0.95) ← **wrong!**
- `domain = "com"` → matches `"reuters.com"` (score 0.97) ← **wrong!**
- `domain = "fakenews.bbc.com"` → would match `"bbc.com"` ← correct, but the first direction handles this

**Fix:** Remove the reverse check, or require a `.` boundary:
```diff
  for known_domain, score in TRUSTED_SOURCES.items():
-     if domain.endswith(known_domain) or known_domain.endswith(domain):
+     if domain.endswith("." + known_domain) or domain == known_domain:
          return score
```

---

### 3. SBERT Model Loads at Import Time — Blocks Startup

**File:** [similarity_checker.py, lines 22-29](file:///c:/Users/VANDIT%20SHARMA/OneDrive/Desktop/CollegeProject/vandit/fake-news-verifier/src/verification/similarity_checker.py#L22-L29)

```python
_SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # runs at module import!
```

This **downloads and loads a 90MB model into RAM the moment any module imports `similarity_checker`** — even if the user only clicks "Quick Predict" (which doesn't need it). It also blocks app startup for 5-15 seconds.

**Fix:** Lazy-load on first use:
```python
def _get_sbert_model():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT_MODEL
```

---

## 🟠 Performance Improvements

### 4. TF-IDF Similarity Creates a NEW Vectorizer Per Pair

**File:** [similarity_checker.py, lines 44-51](file:///c:/Users/VANDIT%20SHARMA/OneDrive/Desktop/CollegeProject/vandit/fake-news-verifier/src/verification/similarity_checker.py#L44-L51)

```python
def _tfidf_similarity(text_a: str, text_b: str) -> float:
    vec = TfidfVectorizer(...)          # NEW vectorizer every call!
    tfidf = vec.fit_transform([text_a, text_b])
```

When checking similarity against 5 sources, this creates **5 separate vectorizers** → 5× vocabulary construction. 

**Fix:** Batch all comparisons together:
```python
def batch_tfidf_similarity(query: str, candidates: List[str]) -> List[float]:
    texts = [query] + candidates
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vec.fit_transform(texts)
    # cosine similarity of query (row 0) vs each candidate
    scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    return scores.tolist()
```

---

### 5. Sequential News Searches — Should Be Parallelized

**File:** [fact_verifier.py, lines 204-207](file:///c:/Users/VANDIT%20SHARMA/OneDrive/Desktop/CollegeProject/vandit/fake-news-verifier/src/verification/fact_verifier.py#L204-L207)

```python
for claim in claims:  # 3 claims × 12sec timeout each = 36 seconds worst case!
    hits = self._search(claim, api_key=api_key, max_results=self.max_sources)
```

Each claim search waits for the previous one to finish. With 3 claims and 15-second timeout, worst case is **45 seconds**.

**Fix:** Use `concurrent.futures.ThreadPoolExecutor`:
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=3) as pool:
    futures = {pool.submit(self._search, claim, api_key=api_key, 
               max_results=self.max_sources): claim for claim in claims}
    for future in as_completed(futures):
        all_articles.extend(future.result())
```

---

### 6. Streamlit Cache Invalidated on Every Slider Change

**File:** [streamlit_app.py, lines 217-230](file:///c:/Users/VANDIT%20SHARMA/OneDrive/Desktop/CollegeProject/vandit/fake-news-verifier/frontend/streamlit_app.py#L217-L230)

```python
@st.cache_resource(show_spinner="Loading ML model…")
def load_verifier(w_ml, w_sim, w_cred, threshold, max_sources):  # ← ALL args are cache keys!
```

Moving **any** slider (weight, threshold, max_sources) **reloads the entire ML model from disk**. The model doesn't change — only the parameters do.

**Fix:** Cache the verifier with no args, set parameters at runtime:
```python
@st.cache_resource(show_spinner="Loading ML model…")
def load_verifier():
    return FactVerifier(pipeline_path=..., model_path=..., vectorizer_path=...)

verifier = load_verifier()
verifier.threshold = threshold  # set per-run, doesn't bust cache
```

---

### 7. No Response Caching for Identical Queries

The same article text searched multiple times hits the external APIs every time. Add an LRU cache:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def _cached_search(query_hash: str, query: str, api_key: str, max_results: int):
    return search_news(query, api_key=api_key, max_results=max_results)
```

---

## 🟡 Accuracy & ML Improvements

### 8. Data Leakage in Training Data

**File:** [train_model.py, lines 84-88](file:///c:/Users/VANDIT%20SHARMA/OneDrive/Desktop/CollegeProject/vandit/fake-news-verifier/train_model.py#L84-L88)

The 99.75% accuracy is suspiciously high. The Kaggle `True.csv`/`Fake.csv` dataset has known issues:
- **Real articles** contain Reuters/AP attribution lines (e.g., `"(Reuters) -"`) that the model memorizes as "REAL" markers
- **Fake articles** often contain domain-specific patterns

The model is detecting **source attribution**, not fake news.

**Fix — strip source markers before training:**
```python
def strip_source_markers(text: str) -> str:
    """Remove Reuters/AP/etc. attribution that leaks the label."""
    text = re.sub(r'\(Reuters\)\s*[-–—]?\s*', '', text)
    text = re.sub(r'\(AP\)\s*[-–—]?\s*', '', text)
    text = re.sub(r'^[A-Z]{2,}[\s,]+\(.*?\)\s*[-–—]\s*', '', text)  # "WASHINGTON (Reuters) -"
    return text.strip()
```

### 9. Only Unigrams — Missing Important Bigram Patterns

**File:** [train_model.py, line 115](file:///c:/Users/VANDIT%20SHARMA/OneDrive/Desktop/CollegeProject/vandit/fake-news-verifier/train_model.py#L115)

`ngram_range=(1, 1)` means the model can't capture phrases like "breaking news", "sources say", "experts claim", "according to" — which are strong signals.

**Fix:**
```diff
- ngram_range=(1, 1),   # unigrams only
+ ngram_range=(1, 2),   # unigrams + bigrams
+ max_features=15_000,  # increase to accommodate bigrams
```

### 10. Similarity Measures Average — Should Use Max

**File:** [fact_verifier.py, lines 230-232](file:///c:/Users/VANDIT%20SHARMA/OneDrive/Desktop/CollegeProject/vandit/fake-news-verifier/src/verification/fact_verifier.py#L230-L232)

```python
source_similarity = float(sum(sim_scores) / len(sim_scores))  # average
```

If 1 out of 5 sources strongly confirms the claim (sim = 0.9) but 4 are irrelevant (sim = 0.1), the average is **0.26** — falsely suggesting weak corroboration. You should use **max** (best matching source) or **top-k average**:

```diff
- source_similarity = float(sum(sim_scores) / len(sim_scores))
+ # Use average of top-2 to reduce noise from irrelevant results
+ sorted_scores = sorted(sim_scores, reverse=True)
+ top_k = sorted_scores[:min(2, len(sorted_scores))]
+ source_similarity = float(sum(top_k) / len(top_k))
```

### 11. No Contradiction Detection

Currently, the system only checks **"do other sources say similar things?"** but never checks **"do other sources contradict this?"**. A claim like _"NASA says Earth is flat"_ might get high similarity because many articles mention "NASA" and "Earth" — but they're all **debunking** it.

**Fix (basic):** Add a contradiction signal by checking if source articles contain negation near key claim terms, or add a simple entailment classifier (e.g., with a cross-encoder model).

---

## 🔵 Feature Additions

### 12. Add Confidence Level to Output

Instead of just "REAL" or "FAKE", provide a confidence band:

```python
if final_score >= 0.75:
    confidence = "HIGH"
elif final_score >= threshold:
    confidence = "MODERATE"
elif final_score >= threshold - 0.15:
    confidence = "LOW"
else:
    confidence = "HIGH"

# Result: "REAL (High Confidence)" or "FAKE (Low Confidence)"
```

### 13. Add Request Timeout & Rate Limiting to FastAPI

```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests=30, window=60):
        super().__init__(app)
        self.requests = {}
        self.max_requests = max_requests
        self.window = window

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()
        # ... rate limiting logic
```

### 14. Add Verification History / Logging

Save each verification to a local JSON log for user to review later:

```python
def _log_verification(result: VerificationResult, article_text: str):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input_preview": article_text[:200],
        **result.to_dict()
    }
    log_path = ROOT / "outputs" / "logs" / "verification_history.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
```

### 15. Add Input Language Detection

Currently only works for English. Add a language check and warn users:

```python
try:
    from langdetect import detect
    lang = detect(article_text)
    if lang != "en":
        st.warning(f"⚠️ Article appears to be in '{lang}'. Results may be less accurate (model trained on English).")
except ImportError:
    pass
```

---

## 📊 Summary — Priority Matrix

| # | Issue | Priority | Effort | Impact |
|---|-------|----------|--------|--------|
| 1 | Race condition in `/verify` | 🔴 Critical | 30 min | Prevents data corruption |
| 2 | False positive domain matching | 🔴 Critical | 10 min | Prevents wrong credibility scores |
| 3 | SBERT blocks startup | 🔴 Critical | 15 min | 5-15s faster startup |
| 4 | TF-IDF vectorizer per pair | 🟠 High | 30 min | ~3-5× faster similarity |
| 5 | Sequential claim searches | 🟠 High | 20 min | Up to 3× faster verification |
| 6 | Streamlit cache busted by sliders | 🟠 High | 15 min | Stops unnecessary model reloads |
| 7 | No response caching | 🟠 Medium | 20 min | Saves API quota + time |
| 8 | Data leakage (source markers) | 🟡 High | 1 hour | More honest model accuracy |
| 9 | Unigrams only | 🟡 Medium | 15 min | Better ML feature capture |
| 10 | Average vs max similarity | 🟡 Medium | 10 min | Better corroboration signal |
| 11 | No contradiction detection | 🟡 High | 3+ hours | Major accuracy improvement |
| 12 | Confidence levels | 🔵 Low | 20 min | Better UX |
| 13 | Rate limiting | 🔵 Low | 30 min | Production safety |
| 14 | Verification history | 🔵 Low | 30 min | User convenience |
| 15 | Language detection | 🔵 Low | 10 min | Better UX for non-English |

> [!TIP]
> **Recommended order:** Fix items 1-3 first (bugs), then 4-6 (performance), then 8-10 (accuracy). Items 12-15 are nice-to-haves for polish.
