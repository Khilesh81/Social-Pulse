"""
sentiment_model.py — Sentiment Analysis using TF-IDF + Logistic Regression.
Labels are auto-generated based on keyword heuristics for training data.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from queries import get_all_posts_for_ml, get_all_comments


# ── Keyword-based label generation ────────────────────────────────────────────

POSITIVE_WORDS = {
    "amazing", "great", "excellent", "awesome", "fantastic", "love",
    "incredible", "brilliant", "insightful", "helpful", "congrats",
    "wonderful", "superb", "exciting", "thank", "thanks", "perfect",
    "outstanding", "impressed", "success", "achieved", "best",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "hate", "useless",
    "disappointing", "worst", "broken", "failed", "wrong", "bug",
    "error", "issue", "problem", "slow", "crash", "disagree",
    "unfortunately", "annoying", "frustrating", "waste",
}


def auto_label_sentiment(text: str) -> str:
    """Heuristic rule-based sentiment labeler for training data generation."""
    if not isinstance(text, str):
        return "neutral"
    words = set(text.lower().split())
    pos_score = len(words & POSITIVE_WORDS)
    neg_score = len(words & NEGATIVE_WORDS)
    if pos_score > neg_score:
        return "positive"
    elif neg_score > pos_score:
        return "negative"
    else:
        return "neutral"


# ── Training ──────────────────────────────────────────────────────────────────

def train_sentiment_model():
    """
    Train TF-IDF + Logistic Regression sentiment classifier.
    Uses auto-labeled post + comment content.
    Returns: (vectorizer, model, accuracy, label_map, report_str)
    """
    posts_df    = get_all_posts_for_ml()[["content"]].rename(columns={"content": "text"})
    comments_df = get_all_comments()[["content"]].rename(columns={"content": "text"})

    df = pd.concat([posts_df, comments_df], ignore_index=True)
    df = df.dropna(subset=["text"])
    df["label"] = df["text"].apply(auto_label_sentiment)

    # Encode labels
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["label_enc"] = df["label"].map(label_map)

    X = df["text"]
    y = df["label_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=500,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train_tfidf, y_train)

    y_pred   = model.predict(X_test_tfidf)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    report   = classification_report(
        y_test, y_pred,
        target_names=["Negative", "Neutral", "Positive"]
    )

    return vectorizer, model, accuracy, label_map, report


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_sentiment(vectorizer, model, text: str) -> dict:
    """Predict sentiment label + probabilities for a given text."""
    vec  = vectorizer.transform([text])
    prob = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]

    label_names = {0: "😠 Negative", 1: "😐 Neutral", 2: "😊 Positive"}
    return {
        "label":    label_names[pred],
        "negative": round(prob[0] * 100, 1),
        "neutral":  round(prob[1] * 100, 1),
        "positive": round(prob[2] * 100, 1),
    }


def get_bulk_sentiment(vectorizer, model, df: pd.DataFrame, text_col: str = "content") -> pd.DataFrame:
    """Predict sentiment for an entire DataFrame column, returns df with 'sentiment' column."""
    texts  = df[text_col].fillna("").tolist()
    vecs   = vectorizer.transform(texts)
    preds  = model.predict(vecs)
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    df = df.copy()
    df["sentiment"] = [labels[p] for p in preds]
    return df
