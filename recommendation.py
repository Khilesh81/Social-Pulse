"""
recommendation.py — User Recommendation System using TF-IDF + Cosine Similarity.
Recommends users based on shared interests and engagement patterns.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from queries import get_user_profiles


# ── Build User Feature Matrix ─────────────────────────────────────────────────

def build_user_matrix(df: pd.DataFrame):
    """
    Combine user interests + activity into a text feature for TF-IDF.
    Returns (vectorizer, tfidf_matrix, df)
    """
    df = df.copy()
    df["interests"] = df["interests"].fillna("general tech")

    def _build_row(row):
        interests_text  = str(row["interests"]).replace(",", " ")
        posts_text      = "post "    * min(int(row["posts_made"]),    50)
        likes_text      = "like "    * min(int(row["likes_given"]),   50)
        comments_text   = "comment " * min(int(row["comments_made"]), 50)
        return f"{interests_text} {posts_text.strip()} {likes_text.strip()} {comments_text.strip()}"

    df["profile_text"] = df.apply(_build_row, axis=1)

    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 1),
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(df["profile_text"])

    return vectorizer, tfidf_matrix, df


def get_recommendations(username: str, top_n: int = 5) -> pd.DataFrame:
    """
    Return top_n recommended users for the given username,
    sorted by cosine similarity of their interest/engagement profiles.
    """
    df = get_user_profiles()

    if username not in df["username"].values:
        return pd.DataFrame(columns=["username", "interests", "similarity_score"])

    _, tfidf_matrix, df = build_user_matrix(df)

    user_idx = df.index[df["username"] == username].tolist()[0]
    user_vec = tfidf_matrix[user_idx]

    # Compute cosine similarity with all other users
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    similarities[user_idx] = -1  # Exclude self

    top_indices = similarities.argsort()[::-1][:top_n]

    result = df.iloc[top_indices][["username", "interests", "posts_made", "likes_given"]].copy()
    result["similarity_score"] = (similarities[top_indices] * 100).round(1)

    return result.reset_index(drop=True)


def get_similar_users_matrix(top_users: int = 20) -> pd.DataFrame:
    """Return a similarity heatmap DataFrame for the top N users."""
    df = get_user_profiles().head(top_users)
    _, tfidf_matrix, df = build_user_matrix(df)

    sim_matrix = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(
        sim_matrix,
        index=df["username"].values,
        columns=df["username"].values
    )
