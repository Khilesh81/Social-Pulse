"""
ml_model.py — Post Engagement Prediction using RandomForestClassifier.
Cached with st.cache_resource so it only trains once per session.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from queries import get_all_posts_for_ml


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for engagement prediction."""
    df = df.copy()
    df["content_length"] = df["content"].str.len()
    df["word_count"]     = df["content"].str.split().str.len()
    df["has_hashtag"]    = df["content"].str.contains("#").astype(int)
    df["has_question"]   = df["content"].str.contains("\\?").astype(int)
    df["has_exclaim"]    = df["content"].str.contains("!").astype(int)
    df["has_link"]       = df["content"].str.contains("http|www", case=False).astype(int)

    # Hour of day
    df["hour"] = pd.to_datetime(df["created_at"]).dt.hour

    # Media type encoding
    le = LabelEncoder()
    df["media_encoded"] = le.fit_transform(df["media_type"].fillna("text"))

    return df


def train_engagement_model():
    """
    Train a RandomForestClassifier to predict if a post will have
    high engagement (above median total_engagement).
    Returns: (model, accuracy, feature_names, report_str)
    """
    df = get_all_posts_for_ml()
    df = build_features(df)

    # Binary label: 1 = high engagement, 0 = low engagement
    median_eng = df["total_engagement"].median()
    df["high_engagement"] = (df["total_engagement"] >= median_eng).astype(int)

    feature_cols = [
        "content_length", "word_count", "has_hashtag", "has_question",
        "has_exclaim", "has_link", "hour", "media_encoded"
    ]

    X = df[feature_cols]
    y = df["high_engagement"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    report   = classification_report(y_test, y_pred, target_names=["Low", "High"])

    return model, accuracy, feature_cols, report


def predict_engagement(model, content: str, media_type: str = "text", hour: int = 12) -> dict:
    """Predict engagement for a new post."""
    content_length = len(content)
    word_count     = len(content.split())
    has_hashtag    = int("#" in content)
    has_question   = int("?" in content)
    has_exclaim    = int("!" in content)
    has_link       = int("http" in content.lower() or "www" in content.lower())

    media_map = {"text": 0, "image": 1, "link": 2, "video": 3}
    media_encoded = media_map.get(media_type, 0)

    X_new = pd.DataFrame([{
        "content_length": content_length,
        "word_count":     word_count,
        "has_hashtag":    has_hashtag,
        "has_question":   has_question,
        "has_exclaim":    has_exclaim,
        "has_link":       has_link,
        "hour":           hour,
        "media_encoded":  media_encoded,
    }])

    prob   = model.predict_proba(X_new)[0]
    label  = model.predict(X_new)[0]

    return {
        "prediction": "🔥 High Engagement" if label == 1 else "📉 Low Engagement",
        "confidence": round(max(prob) * 100, 1),
        "prob_high":  round(prob[1] * 100, 1),
        "prob_low":   round(prob[0] * 100, 1),
    }


def get_feature_importances(model, feature_cols: list) -> pd.DataFrame:
    """Return a DataFrame of feature importances for visualization."""
    importances = model.feature_importances_
    df = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    return df
