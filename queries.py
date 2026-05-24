"""
queries.py — All SQL analytics queries, wrapped with Pandas for display.
Uses st.cache_data via caller in app.py.
"""

import sqlite3
import pandas as pd
from database import get_connection


# ── KPI Metrics ───────────────────────────────────────────────────────────────

def get_kpis() -> dict:
    conn = get_connection()
    cur = conn.cursor()

    total_users    = cur.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_posts    = cur.execute("SELECT COUNT(*) FROM posts WHERE is_spam=0").fetchone()[0]
    total_comments = cur.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    total_likes    = cur.execute("SELECT COUNT(*) FROM likes").fetchone()[0]
    total_shares   = cur.execute("SELECT COUNT(*) FROM shares").fetchone()[0]

    # Engagement rate = (likes + comments + shares) / posts
    engagement_rate = round(
        (total_likes + total_comments + total_shares) / max(total_posts, 1), 2
    )

    # Most active community
    row = cur.execute("""
        SELECT c.name, COUNT(p.post_id) as post_count
        FROM communities c
        JOIN posts p ON c.community_id = p.community_id
        WHERE p.is_spam = 0
        GROUP BY c.community_id
        ORDER BY post_count DESC LIMIT 1
    """).fetchone()
    most_active_community = row[0] if row else "N/A"

    conn.close()
    return {
        "total_users": total_users,
        "total_posts": total_posts,
        "total_comments": total_comments,
        "total_likes": total_likes,
        "total_shares": total_shares,
        "engagement_rate": engagement_rate,
        "most_active_community": most_active_community,
    }


# ── Top Users ─────────────────────────────────────────────────────────────────

def get_top_users(limit: int = 10) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            u.username,
            COUNT(DISTINCT p.post_id)   AS posts,
            COUNT(DISTINCT l.like_id)   AS likes_received,
            COUNT(DISTINCT c.comment_id) AS comments_received,
            COUNT(DISTINCT s.share_id)  AS shares_received,
            (COUNT(DISTINCT l.like_id) + COUNT(DISTINCT c.comment_id) + COUNT(DISTINCT s.share_id)) AS total_engagement
        FROM users u
        LEFT JOIN posts    p ON u.user_id = p.user_id AND p.is_spam = 0
        LEFT JOIN likes    l ON p.post_id = l.post_id
        LEFT JOIN comments c ON p.post_id = c.post_id
        LEFT JOIN shares   s ON p.post_id = s.post_id
        GROUP BY u.user_id
        ORDER BY total_engagement DESC
        LIMIT ?
    """, conn, params=(limit,))
    conn.close()
    return df


# ── Trending Hashtags ─────────────────────────────────────────────────────────

def get_trending_hashtags(limit: int = 15) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT h.tag AS hashtag, COUNT(*) AS usage_count
        FROM hashtags h
        JOIN posts p ON h.post_id = p.post_id
        WHERE p.is_spam = 0
        GROUP BY h.tag
        ORDER BY usage_count DESC
        LIMIT ?
    """, conn, params=(limit,))
    conn.close()
    return df


# ── Most Liked Posts ──────────────────────────────────────────────────────────

def get_most_liked_posts(limit: int = 10) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            p.post_id,
            u.username,
            SUBSTR(p.content, 1, 80) || '...' AS preview,
            COUNT(l.like_id) AS likes,
            p.created_at
        FROM posts p
        JOIN users u ON p.user_id = u.user_id
        LEFT JOIN likes l ON p.post_id = l.post_id
        WHERE p.is_spam = 0
        GROUP BY p.post_id
        ORDER BY likes DESC
        LIMIT ?
    """, conn, params=(limit,))
    conn.close()
    return df


# ── Most Commented Posts ──────────────────────────────────────────────────────

def get_most_commented_posts(limit: int = 10) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            p.post_id,
            u.username,
            SUBSTR(p.content, 1, 80) || '...' AS preview,
            COUNT(c.comment_id) AS comments,
            p.created_at
        FROM posts p
        JOIN users u ON p.user_id = u.user_id
        LEFT JOIN comments c ON p.post_id = c.post_id
        WHERE p.is_spam = 0
        GROUP BY p.post_id
        ORDER BY comments DESC
        LIMIT ?
    """, conn, params=(limit,))
    conn.close()
    return df


# ── Community Engagement ──────────────────────────────────────────────────────

def get_community_engagement() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            c.name AS community,
            COUNT(DISTINCT p.post_id)    AS posts,
            COUNT(DISTINCT l.like_id)    AS likes,
            COUNT(DISTINCT cm.comment_id) AS comments,
            COUNT(DISTINCT s.share_id)   AS shares,
            c.member_count
        FROM communities c
        LEFT JOIN posts    p  ON c.community_id = p.community_id AND p.is_spam = 0
        LEFT JOIN likes    l  ON p.post_id = l.post_id
        LEFT JOIN comments cm ON p.post_id = cm.post_id
        LEFT JOIN shares   s  ON p.post_id = s.post_id
        GROUP BY c.community_id
        ORDER BY posts DESC
    """, conn)
    conn.close()
    return df


# ── Engagement Trend (daily) ──────────────────────────────────────────────────

def get_engagement_trend() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            DATE(p.created_at) AS date,
            COUNT(DISTINCT p.post_id)    AS posts,
            COUNT(DISTINCT l.like_id)    AS likes,
            COUNT(DISTINCT c.comment_id) AS comments,
            COUNT(DISTINCT s.share_id)   AS shares
        FROM posts p
        LEFT JOIN likes    l ON p.post_id = l.post_id
        LEFT JOIN comments c ON p.post_id = c.post_id
        LEFT JOIN shares   s ON p.post_id = s.post_id
        WHERE p.is_spam = 0
        GROUP BY DATE(p.created_at)
        ORDER BY date ASC
    """, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── User Activity Heatmap ─────────────────────────────────────────────────────

def get_activity_heatmap() -> pd.DataFrame:
    """Returns post counts by day-of-week and hour-of-day."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            CAST(strftime('%w', created_at) AS INTEGER) AS day_of_week,
            CAST(strftime('%H', created_at) AS INTEGER) AS hour_of_day,
            COUNT(*) AS post_count
        FROM posts
        WHERE is_spam = 0
        GROUP BY day_of_week, hour_of_day
    """, conn)
    conn.close()
    pivot = df.pivot_table(
        index="day_of_week", columns="hour_of_day", values="post_count", fill_value=0
    )
    days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    pivot.index = [days[i] for i in pivot.index]
    return pivot


# ── All Posts (for ML input) ──────────────────────────────────────────────────

def get_all_posts_for_ml() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            p.post_id,
            p.user_id,
            p.community_id,
            p.content,
            p.media_type,
            p.created_at,
            p.is_spam,
            COUNT(DISTINCT l.like_id)    AS likes,
            COUNT(DISTINCT c.comment_id) AS comments,
            COUNT(DISTINCT s.share_id)   AS shares
        FROM posts p
        LEFT JOIN likes    l ON p.post_id = l.post_id
        LEFT JOIN comments c ON p.post_id = c.post_id
        LEFT JOIN shares   s ON p.post_id = s.post_id
        GROUP BY p.post_id
    """, conn)
    conn.close()
    df["total_engagement"] = df["likes"] + df["comments"] + df["shares"]
    return df


# ── All Comments (for sentiment) ─────────────────────────────────────────────

def get_all_comments() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT c.comment_id, c.content, c.created_at, u.username
        FROM comments c
        JOIN users u ON c.user_id = u.user_id
    """, conn)
    conn.close()
    return df


# ── User Profiles (for recommendation) ───────────────────────────────────────

def get_user_profiles() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            u.user_id,
            u.username,
            u.interests,
            COUNT(DISTINCT p.post_id)    AS posts_made,
            COUNT(DISTINCT l.like_id)    AS likes_given,
            COUNT(DISTINCT c.comment_id) AS comments_made
        FROM users u
        LEFT JOIN posts    p ON u.user_id = p.user_id
        LEFT JOIN likes    l ON u.user_id = l.user_id
        LEFT JOIN comments c ON u.user_id = c.user_id
        GROUP BY u.user_id
    """, conn)
    conn.close()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH & ANALYZE QUERIES
# ══════════════════════════════════════════════════════════════════════════════

# ── User profile deep-dive ────────────────────────────────────────────────────

def search_user_profile(username: str) -> dict | None:
    """Return a dict with full profile stats for a given username."""
    conn = get_connection()
    cur  = conn.cursor()
    row  = cur.execute("""
        SELECT u.user_id, u.username, u.email, u.bio, u.interests,
               u.joined_at, u.is_active
        FROM users u WHERE u.username = ?
    """, (username,)).fetchone()

    if row is None:
        conn.close()
        return None

    uid = row["user_id"]

    posts_count    = cur.execute("SELECT COUNT(*) FROM posts    WHERE user_id=? AND is_spam=0", (uid,)).fetchone()[0]
    likes_received = cur.execute("SELECT COUNT(*) FROM likes    l JOIN posts p ON l.post_id=p.post_id WHERE p.user_id=?", (uid,)).fetchone()[0]
    comments_recv  = cur.execute("SELECT COUNT(*) FROM comments c JOIN posts p ON c.post_id=p.post_id WHERE p.user_id=?", (uid,)).fetchone()[0]
    shares_recv    = cur.execute("SELECT COUNT(*) FROM shares   s JOIN posts p ON s.post_id=p.post_id WHERE p.user_id=?", (uid,)).fetchone()[0]
    followers_cnt  = cur.execute("SELECT COUNT(*) FROM followers WHERE following_id=?", (uid,)).fetchone()[0]
    following_cnt  = cur.execute("SELECT COUNT(*) FROM followers WHERE follower_id=?",  (uid,)).fetchone()[0]

    conn.close()
    return {
        "user_id":        uid,
        "username":       row["username"],
        "email":          row["email"],
        "bio":            row["bio"] or "—",
        "interests":      row["interests"] or "—",
        "joined_at":      row["joined_at"],
        "is_active":      bool(row["is_active"]),
        "posts_count":    posts_count,
        "likes_received": likes_received,
        "comments_recv":  comments_recv,
        "shares_recv":    shares_recv,
        "total_engagement": likes_received + comments_recv + shares_recv,
        "followers":      followers_cnt,
        "following":      following_cnt,
    }


def get_user_posts(username: str, limit: int = 15) -> pd.DataFrame:
    """Return posts by a specific user with engagement counts."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            p.post_id,
            SUBSTR(p.content, 1, 100) || '...' AS preview,
            p.media_type,
            p.created_at,
            COUNT(DISTINCT l.like_id)    AS likes,
            COUNT(DISTINCT c.comment_id) AS comments,
            COUNT(DISTINCT s.share_id)   AS shares,
            (COUNT(DISTINCT l.like_id) + COUNT(DISTINCT c.comment_id) + COUNT(DISTINCT s.share_id)) AS engagement
        FROM posts p
        JOIN users u ON p.user_id = u.user_id
        LEFT JOIN likes    l ON p.post_id = l.post_id
        LEFT JOIN comments c ON p.post_id = c.post_id
        LEFT JOIN shares   s ON p.post_id = s.post_id
        WHERE u.username = ? AND p.is_spam = 0
        GROUP BY p.post_id
        ORDER BY engagement DESC
        LIMIT ?
    """, conn, params=(username, limit))
    conn.close()
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def get_user_activity_timeline(username: str) -> pd.DataFrame:
    """Monthly post activity for a specific user."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            strftime('%Y-%m', p.created_at) AS month,
            COUNT(p.post_id)                AS posts
        FROM posts p
        JOIN users u ON p.user_id = u.user_id
        WHERE u.username = ? AND p.is_spam = 0
        GROUP BY month ORDER BY month ASC
    """, conn, params=(username,))
    conn.close()
    return df


# ── Hashtag deep-dive ─────────────────────────────────────────────────────────

def search_hashtag_detail(tag: str) -> dict:
    """Return aggregate stats for a hashtag."""
    tag = tag.lstrip("#").lower().strip()
    conn = get_connection()
    cur  = conn.cursor()

    usage = cur.execute("""
        SELECT COUNT(*) FROM hashtags h
        JOIN posts p ON h.post_id = p.post_id
        WHERE h.tag = ? AND p.is_spam = 0
    """, (tag,)).fetchone()[0]

    conn.close()
    return {"tag": tag, "usage_count": usage}


def get_hashtag_posts(tag: str, limit: int = 15) -> pd.DataFrame:
    """Return posts containing a specific hashtag."""
    tag = tag.lstrip("#").lower().strip()
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            u.username,
            SUBSTR(p.content, 1, 120) || '...' AS preview,
            p.media_type,
            p.created_at,
            COUNT(DISTINCT l.like_id)    AS likes,
            COUNT(DISTINCT c.comment_id) AS comments,
            COUNT(DISTINCT s.share_id)   AS shares,
            (COUNT(DISTINCT l.like_id) + COUNT(DISTINCT c.comment_id) + COUNT(DISTINCT s.share_id)) AS engagement
        FROM hashtags h
        JOIN posts    p ON h.post_id  = p.post_id
        JOIN users    u ON p.user_id  = u.user_id
        LEFT JOIN likes    l ON p.post_id = l.post_id
        LEFT JOIN comments c ON p.post_id = c.post_id
        LEFT JOIN shares   s ON p.post_id = s.post_id
        WHERE h.tag = ? AND p.is_spam = 0
        GROUP BY p.post_id
        ORDER BY engagement DESC
        LIMIT ?
    """, conn, params=(tag, limit))
    conn.close()
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def get_hashtag_trend(tag: str) -> pd.DataFrame:
    """Monthly usage trend for a hashtag."""
    tag = tag.lstrip("#").lower().strip()
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            strftime('%Y-%m', p.created_at) AS month,
            COUNT(h.hashtag_id)             AS usage
        FROM hashtags h
        JOIN posts p ON h.post_id = p.post_id
        WHERE h.tag = ? AND p.is_spam = 0
        GROUP BY month ORDER BY month ASC
    """, conn, params=(tag,))
    conn.close()
    return df


def get_hashtag_top_users(tag: str, limit: int = 10) -> pd.DataFrame:
    """Top users who use a specific hashtag."""
    tag = tag.lstrip("#").lower().strip()
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT u.username, COUNT(*) AS post_count
        FROM hashtags h
        JOIN posts p ON h.post_id = p.post_id
        JOIN users u ON p.user_id = u.user_id
        WHERE h.tag = ? AND p.is_spam = 0
        GROUP BY u.user_id
        ORDER BY post_count DESC
        LIMIT ?
    """, conn, params=(tag, limit))
    conn.close()
    return df


# ── Community deep-dive ───────────────────────────────────────────────────────

def search_community_detail(name: str) -> dict | None:
    """Return aggregate stats for a community by name (partial match)."""
    conn = get_connection()
    cur  = conn.cursor()
    row  = cur.execute("""
        SELECT community_id, name, description, created_at, member_count
        FROM communities
        WHERE name LIKE ?
        LIMIT 1
    """, (f"%{name}%",)).fetchone()

    if row is None:
        conn.close()
        return None

    cid = row["community_id"]
    posts    = cur.execute("SELECT COUNT(*) FROM posts WHERE community_id=? AND is_spam=0", (cid,)).fetchone()[0]
    likes    = cur.execute("SELECT COUNT(l.like_id) FROM likes l JOIN posts p ON l.post_id=p.post_id WHERE p.community_id=?", (cid,)).fetchone()[0]
    comments = cur.execute("SELECT COUNT(c.comment_id) FROM comments c JOIN posts p ON c.post_id=p.post_id WHERE p.community_id=?", (cid,)).fetchone()[0]
    shares   = cur.execute("SELECT COUNT(s.share_id) FROM shares s JOIN posts p ON s.post_id=p.post_id WHERE p.community_id=?", (cid,)).fetchone()[0]
    conn.close()

    return {
        "community_id":  cid,
        "name":          row["name"],
        "description":   row["description"],
        "created_at":    row["created_at"],
        "member_count":  row["member_count"],
        "total_posts":   posts,
        "total_likes":   likes,
        "total_comments":comments,
        "total_shares":  shares,
        "total_engagement": likes + comments + shares,
    }


def get_community_top_contributors(community_name: str, limit: int = 10) -> pd.DataFrame:
    """Top posting users in a community (partial name match)."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            u.username,
            COUNT(DISTINCT p.post_id)    AS posts,
            COUNT(DISTINCT l.like_id)    AS likes_received,
            COUNT(DISTINCT cm.comment_id) AS comments_received,
            (COUNT(DISTINCT l.like_id) + COUNT(DISTINCT cm.comment_id)) AS engagement
        FROM users u
        JOIN posts      p  ON u.user_id = p.user_id
        JOIN communities c  ON p.community_id = c.community_id
        LEFT JOIN likes    l  ON p.post_id = l.post_id
        LEFT JOIN comments cm ON p.post_id = cm.post_id
        WHERE c.name LIKE ? AND p.is_spam = 0
        GROUP BY u.user_id
        ORDER BY posts DESC
        LIMIT ?
    """, conn, params=(f"%{community_name}%", limit))
    conn.close()
    return df


def get_community_posts(community_name: str, limit: int = 10) -> pd.DataFrame:
    """Top posts from a community."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            u.username,
            SUBSTR(p.content, 1, 100) || '...' AS preview,
            p.media_type, p.created_at,
            COUNT(DISTINCT l.like_id)    AS likes,
            COUNT(DISTINCT cm.comment_id) AS comments,
            (COUNT(DISTINCT l.like_id) + COUNT(DISTINCT cm.comment_id)) AS engagement
        FROM posts p
        JOIN users      u  ON p.user_id = u.user_id
        JOIN communities c  ON p.community_id = c.community_id
        LEFT JOIN likes    l  ON p.post_id = l.post_id
        LEFT JOIN comments cm ON p.post_id = cm.post_id
        WHERE c.name LIKE ? AND p.is_spam = 0
        GROUP BY p.post_id
        ORDER BY engagement DESC
        LIMIT ?
    """, conn, params=(f"%{community_name}%", limit))
    conn.close()
    return df
