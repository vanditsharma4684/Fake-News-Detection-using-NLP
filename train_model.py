#!/usr/bin/env python3
"""
train_model.py
--------------
Train TF-IDF + LogisticRegression pipeline on the fake/real news dataset.

Usage:
    python train_model.py \
        --real  data/raw/True.csv \
        --fake  data/raw/Fake.csv \
        --outdir models
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.preprocessing.text_clean import clean_many
from src.utils.helpers import ensure_dir, save_json

LABELS = ("REAL", "FAKE")


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def plot_confusion_matrix(cm, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_xticklabels(LABELS)
    ax.set_yticks([0, 1]); ax.set_yticklabels(LABELS)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real",   required=True)
    ap.add_argument("--fake",   required=True)
    ap.add_argument("--outdir", default="models")
    a = ap.parse_args()

    outdir = ensure_dir(Path(a.outdir))
    charts = ensure_dir(outdir / "charts")

    # Load
    df_real = read_csv(Path(a.real))
    df_fake = read_csv(Path(a.fake))

    # Build combined text
    for df in (df_real, df_fake):
        title = df.get("title", pd.Series([""] * len(df))).fillna("")
        text  = df.get("text",  pd.Series([""] * len(df))).fillna("")
        df["combined"] = (title + " " + text).str.strip()

    X_raw = pd.concat([df_real["combined"], df_fake["combined"]], ignore_index=True)
    y     = np.array([0] * len(df_real) + [1] * len(df_fake))

    # Clean (crucial: strip source markers to prevent data leakage)
    print("Cleaning text...")
    X = clean_many(X_raw.tolist(), strip_sources=True)

    # ---------------------------------------------------------------
    # Train / test split (80/20) for honest hold-out evaluation
    # ---------------------------------------------------------------
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # ---------------------------------------------------------------
    # Pipeline — memory-safe settings
    #   • unigrams + bigrams (1, 2) captures phrases like "sources say"
    #   • max_features=15_000 caps RAM usage while allowing bigrams
    # ---------------------------------------------------------------
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True,
            stop_words="english",
            ngram_range=(1, 2),   # unigrams + bigrams
            max_features=15_000,  # safe cap for low-RAM machines
            dtype=np.float32,     # halves memory vs float64
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="saga",        # efficient on sparse matrices; single-threaded by design
        )),
    ])

    # ---------------------------------------------------------------
    # Cross-validate on training split only
    #   n_jobs=1  → avoids joblib forking extra copies of the feature matrix
    # ---------------------------------------------------------------
    print("Running 5-fold cross-validation (this may take a minute)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipe, X_train, y_train,
                            cv=cv, scoring="f1_macro", n_jobs=1)
    print(f"CV F1 (5-fold, train split): {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

    # ---------------------------------------------------------------
    # Fit on full training split, evaluate on hold-out test split
    # ---------------------------------------------------------------
    print("Fitting final model on training data...")
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy":     float(accuracy_score(y_test, y_pred)),
        "roc_auc":      float(roc_auc_score(y_test, y_prob)),
        "avg_precision":float(average_precision_score(y_test, y_prob)),
        "cv_f1_mean":   float(cv_f1.mean()),
        "cv_f1_std":    float(cv_f1.std()),
        "report":       classification_report(y_test, y_pred, target_names=LABELS, output_dict=True),
    }

    save_json(metrics, outdir / "metrics.json")
    print("Metrics:", json.dumps({k: metrics[k] for k in ["accuracy", "roc_auc"]}, indent=2))

    # Charts
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, charts / "confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr); ax.set_title("ROC Curve"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    fig.savefig(charts / "roc_curve.png", dpi=150); plt.close(fig)

    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(rec, prec); ax.set_title("Precision-Recall Curve"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    fig.savefig(charts / "pr_curve.png", dpi=150); plt.close(fig)

    # Save artifacts
    joblib.dump(pipe, outdir / "pipeline.joblib")
    joblib.dump(pipe.named_steps["tfidf"], outdir / "vectorizer.joblib")
    joblib.dump(pipe.named_steps["clf"],   outdir / "fake_news_model.joblib")

    print("✅ Training complete. Artifacts saved to:", outdir.resolve())


if __name__ == "__main__":
    main()