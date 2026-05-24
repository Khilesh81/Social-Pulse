"""
database.py — SQLite database initialization, schema creation, and user authentication.
Handles all direct DB connection logic.
"""

import sqlite3
import hashlib
import os
from data_generator import generate_all_data

DB_PATH = "social_verse.db"


def get_connection():
    """Return a SQLite connection with row_factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def hash_password(password: str) -> str:
    """SHA-256 hash of the password."""
    return hashlib.sha256(password.encode()).hexdigest()


def init_db():
    """Create all tables if they don't exist, then seed data if empty."""
    conn = get_connection()
    cur = conn.cursor()

    # ── Users ────────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT UNIQUE NOT NULL,
            email      TEXT UNIQUE NOT NULL,
            password   TEXT NOT NULL,
            bio        TEXT,
            interests  TEXT,
            joined_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active  INTEGER DEFAULT 1
        )
    """)

    # ── Communities ──────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS communities (
            community_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name           TEXT UNIQUE NOT NULL,
            description    TEXT,
            created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            member_count   INTEGER DEFAULT 0
        )
    """)

    # ── Posts ────────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            post_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            community_id  INTEGER,
            content       TEXT NOT NULL,
            media_type    TEXT DEFAULT 'text',
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_spam       INTEGER DEFAULT 0,
            FOREIGN KEY(user_id)      REFERENCES users(user_id),
            FOREIGN KEY(community_id) REFERENCES communities(community_id)
        )
    """)

    # ── Comments ─────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            comment_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id     INTEGER NOT NULL,
            user_id     INTEGER NOT NULL,
            content     TEXT NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(post_id)  REFERENCES posts(post_id),
            FOREIGN KEY(user_id)  REFERENCES users(user_id)
        )
    """)

    # ── Likes ────────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS likes (
            like_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id    INTEGER NOT NULL,
            user_id    INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(post_id, user_id),
            FOREIGN KEY(post_id)  REFERENCES posts(post_id),
            FOREIGN KEY(user_id)  REFERENCES users(user_id)
        )
    """)

    # ── Shares ───────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS shares (
            share_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id    INTEGER NOT NULL,
            user_id    INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(post_id)  REFERENCES posts(post_id),
            FOREIGN KEY(user_id)  REFERENCES users(user_id)
        )
    """)

    # ── Hashtags ─────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hashtags (
            hashtag_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            tag         TEXT NOT NULL,
            post_id     INTEGER NOT NULL,
            FOREIGN KEY(post_id) REFERENCES posts(post_id)
        )
    """)

    # ── Followers ────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS followers (
            follower_id  INTEGER NOT NULL,
            following_id INTEGER NOT NULL,
            followed_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(follower_id, following_id),
            FOREIGN KEY(follower_id)  REFERENCES users(user_id),
            FOREIGN KEY(following_id) REFERENCES users(user_id)
        )
    """)

    conn.commit()

    # Seed sample data only once
    row = cur.execute("SELECT COUNT(*) as c FROM users").fetchone()
    if row["c"] == 0:
        generate_all_data(conn)

    conn.close()


# ── Authentication helpers ────────────────────────────────────────────────────

def register_user(username: str, email: str, password: str, bio: str = "", interests: str = "") -> dict:
    """Register a new user. Returns dict with success/error."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, email, password, bio, interests) VALUES (?,?,?,?,?)",
            (username.strip(), email.strip().lower(), hash_password(password), bio, interests)
        )
        conn.commit()
        return {"success": True}
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return {"success": False, "error": "Username already taken."}
        return {"success": False, "error": "Email already registered."}
    finally:
        conn.close()


def login_user(username: str, password: str):
    """Validate credentials. Returns user row dict or None."""
    conn = get_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username.strip(), hash_password(password))
    ).fetchone()
    conn.close()
    return dict(user) if user else None
