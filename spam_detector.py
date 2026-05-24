"""
spam_detector.py — Spam / Fake Post Detection using TF-IDF + Logistic Regression.
Uses the is_spam flag from the database as ground-truth labels.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from queries import get_all_posts_for_ml


# ── Training ──────────────────────────────────────────────────────────────────

def train_spam_model():
    """
    Train a TF-IDF + Logistic Regression spam detector.
    Uses the `is_spam` column from the posts table as ground truth.
    Returns: (vectorizer, model, accuracy, report_str)
    """
    df = get_all_posts_for_ml()
    df = df.dropna(subset=["content"])

    # Augment spam examples to handle class imbalance
    spam_df = df[df["is_spam"] == 1]
    real_df = df[df["is_spam"] == 0]

    # Upsample spam to balance (simple repeat)
    if len(spam_df) < len(real_df) // 3:
        multiplier = max(1, len(real_df) // (len(spam_df) * 3))
        spam_df = pd.concat([spam_df] * multiplier, ignore_index=True)

    balanced_df = pd.concat([real_df, spam_df], ignore_index=True).sample(
        frac=1, random_state=42
    )

    X = balanced_df["content"]
    y = balanced_df["is_spam"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        analyzer="word",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=300,
        C=2.0,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train_tfidf, y_train)

    y_pred   = model.predict(X_test_tfidf)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    report   = classification_report(
        y_test, y_pred,
        target_names=["Legitimate", "Spam"]
    )
    cm = confusion_matrix(y_test, y_pred)

    return vectorizer, model, accuracy, report, cm


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_spam(vectorizer, model, text: str) -> dict:
    """Predict whether a given text is spam."""
    vec  = vectorizer.transform([text])
    prob = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]

    return {
        "label":       "🚫 Spam / Fake" if pred == 1 else "✅ Legitimate",
        "is_spam":     bool(pred),
        "spam_prob":   round(prob[1] * 100, 1),
        "legit_prob":  round(prob[0] * 100, 1),
        "confidence":  round(max(prob) * 100, 1),
    }


def flag_spam_posts(vectorizer, model, df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with a predicted_spam column added."""
    texts = df["content"].fillna("").tolist()
    vecs  = vectorizer.transform(texts)
    preds = model.predict(vecs)
    df    = df.copy()
    df["predicted_spam"] = preds
    return df
