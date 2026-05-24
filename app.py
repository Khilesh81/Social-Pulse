"""
app.py — AI Social Verse Intelligence Dashboard
Main Streamlit application with authentication, navigation, and all dashboard pages.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ── Project modules ───────────────────────────────────────────────────────────
from database import init_db, login_user, register_user
import queries as Q
from ml_model import train_engagement_model, predict_engagement, get_feature_importances
from sentiment_model import train_sentiment_model, predict_sentiment, get_bulk_sentiment
from spam_detector import train_spam_model, predict_spam
from recommendation import get_recommendations, get_similar_users_matrix

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Social Verse Intelligence Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize DB ─────────────────────────────────────────────────────────────
init_db()

# ═════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — Dark neon theme
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg-primary:    #050a18;
    --bg-secondary:  #080f26;
    --bg-card:       rgba(10, 20, 50, 0.75);
    --border-glow:   rgba(0, 200, 255, 0.25);
    --neon-blue:     #00c8ff;
    --neon-cyan:     #00ffe1;
    --neon-purple:   #a855f7;
    --neon-indigo:   #6366f1;
    --text-primary:  #e8eeff;
    --text-muted:    #7b8db8;
    --accent-grad:   linear-gradient(135deg, #00c8ff 0%, #a855f7 100%);
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Main content background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 10%, rgba(0,100,200,0.10) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 80%, rgba(168,85,247,0.08) 0%, transparent 60%),
                var(--bg-primary);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #040c1e 0%, #060f28 60%, #08142e 100%) !important;
    border-right: 1px solid rgba(0,200,255,0.18) !important;
    box-shadow: 4px 0 30px rgba(0,200,255,0.04) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--neon-blue) !important;
}

/* ── Sidebar nav radio wrapper ── */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 2px !important;
}

/* ── Each nav row ── */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    color: #b0c4e8 !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    padding: 9px 14px !important;
    border-radius: 10px !important;
    cursor: pointer !important;
    transition: background 0.2s ease, color 0.2s ease, transform 0.18s ease !important;
    letter-spacing: 0.2px !important;
    margin: 2px 0 !important;
    border: 1px solid transparent !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(0,200,255,0.09) !important;
    color: #00c8ff !important;
    transform: translateX(5px) !important;
    border-color: rgba(0,200,255,0.15) !important;
}
/* ── Selected item ── */
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {
    background: linear-gradient(135deg, rgba(0,200,255,0.14), rgba(168,85,247,0.10)) !important;
    color: #00c8ff !important;
    font-weight: 700 !important;
    border-color: rgba(0,200,255,0.3) !important;
    box-shadow: 0 0 18px rgba(0,200,255,0.12) !important;
}
/* ── Safely hide only the radio SVG circle ── */
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"] {
    display: none !important;
}

/* ── Animated cursor blink ── */
@keyframes cursorBlink {
    0%, 100% { opacity: 1; }
    45%, 55%  { opacity: 0; }
}
.nav-cursor {
    display: inline-block;
    width: 2px; height: 14px;
    background: #00c8ff;
    margin-left: 6px;
    vertical-align: middle;
    border-radius: 1px;
    animation: cursorBlink 1.1s ease-in-out infinite;
    box-shadow: 0 0 6px #00c8ff;
}

/* ── Sidebar pulse dot ── */
@keyframes pulseDot {
    0%,100% { transform: scale(1);   box-shadow: 0 0 0 0 rgba(0,200,255,0.5); }
    50%      { transform: scale(1.2); box-shadow: 0 0 0 5px rgba(0,200,255,0); }
}
.pulse-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #00c8ff;
    border-radius: 50%;
    animation: pulseDot 2s ease-in-out infinite;
    vertical-align: middle;
    margin-right: 6px;
}

/* ── Sidebar logo glow animation ── */
@keyframes logoFloat {
    0%,100% { filter: drop-shadow(0 0 10px rgba(0,200,255,0.6)); transform: translateY(0px); }
    50%     { filter: drop-shadow(0 0 22px rgba(0,200,255,0.9)); transform: translateY(-4px); }
}
.sidebar-logo { animation: logoFloat 3.5s ease-in-out infinite; display: inline-block; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 0 20px rgba(0,200,255,0.06), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 0 30px rgba(0,200,255,0.15) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--neon-blue) !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00c8ff22, #a855f722) !important;
    color: var(--neon-blue) !important;
    border: 1px solid var(--neon-blue) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 0 14px rgba(0,200,255,0.18) !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00c8ff44, #a855f744) !important;
    box-shadow: 0 0 28px rgba(0,200,255,0.4) !important;
    transform: translateY(-2px) !important;
}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea, .stSelectbox select {
    background: rgba(8,18,46,0.9) !important;
    border: 1px solid rgba(0,200,255,0.3) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--neon-blue) !important;
    box-shadow: 0 0 12px rgba(0,200,255,0.25) !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-glow) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: rgba(8,18,46,0.6) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,200,255,0.2), rgba(168,85,247,0.2)) !important;
    color: var(--neon-blue) !important;
    border: 1px solid rgba(0,200,255,0.3) !important;
}

/* ── Dividers ── */
hr { border-color: rgba(0,200,255,0.15) !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(12px) !important;
}

/* ── Code blocks ── */
code { color: var(--neon-cyan) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #050a18; }
::-webkit-scrollbar-thumb { background: rgba(0,200,255,0.3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,200,255,0.5); }

/* ── Section headings ── */
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Login / Signup page ── */
.auth-container {
    max-width: 420px;
    margin: 4rem auto;
    background: rgba(8,18,46,0.85);
    border: 1px solid rgba(0,200,255,0.3);
    border-radius: 24px;
    padding: 2.5rem 2.8rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 0 60px rgba(0,100,200,0.2), 0 0 120px rgba(168,85,247,0.08);
}
.moon-ring {
    width: 140px; height: 140px;
    border-radius: 50%;
    background: radial-gradient(circle at 38% 38%, #1a2a6c, #0d1b4b 55%, #080f2e 100%);
    box-shadow: 0 0 40px rgba(0,200,255,0.35), 0 0 80px rgba(0,100,200,0.2),
                inset -8px -8px 20px rgba(0,0,0,0.5);
    margin: 0 auto 1.5rem;
    border: 2px solid rgba(0,200,255,0.4);
    display: flex; align-items: center; justify-content: center;
    font-size: 3.2rem;
    animation: moonPulse 4s ease-in-out infinite;
}
@keyframes moonPulse {
    0%,100% { box-shadow: 0 0 40px rgba(0,200,255,0.35), 0 0 80px rgba(0,100,200,0.2); }
    50%      { box-shadow: 0 0 60px rgba(0,200,255,0.55), 0 0 120px rgba(0,100,200,0.35); }
}
.brand-title {
    text-align: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00c8ff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}
.brand-sub {
    text-align: center;
    color: #7b8db8;
    font-size: 0.82rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1.8rem;
}

/* ── KPI card custom ── */
.kpi-card {
    background: rgba(10,20,50,0.8);
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    backdrop-filter: blur(16px);
    text-align: center;
    transition: transform 0.25s, box-shadow 0.25s;
    box-shadow: 0 0 18px rgba(0,200,255,0.05);
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 0 32px rgba(0,200,255,0.18);
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00c8ff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.kpi-label {
    font-size: 0.75rem;
    color: #7b8db8;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin-top: 4px;
}
.kpi-icon { font-size: 1.5rem; margin-bottom: 6px; }

/* ── Section header ── */
.section-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 1.2rem; padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(0,200,255,0.15);
}
.section-header h2 {
    font-size: 1.35rem; margin: 0;
    background: linear-gradient(135deg, #00c8ff, #a855f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}

/* ── Glass info box ── */
.glass-box {
    background: rgba(0,200,255,0.05);
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 12px;
    padding: 1rem 1.3rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* ── Accuracy badge ── */
.acc-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(0,200,255,0.2), rgba(168,85,247,0.2));
    border: 1px solid rgba(0,200,255,0.4);
    border-radius: 24px;
    padding: 4px 18px;
    font-weight: 600;
    color: #00c8ff;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PLOTLY THEME HELPER
# ═════════════════════════════════════════════════════════════════════════════

def dark_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply consistent dark neon theme to any Plotly figure."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(5,10,24,0)",
        plot_bgcolor="rgba(5,10,24,0)",
        font=dict(color="#c8d6f0", family="Inter"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            bgcolor="rgba(8,18,46,0.7)",
            bordercolor="rgba(0,200,255,0.2)",
            borderwidth=1,
            font=dict(color="#c8d6f0"),
        ),
        xaxis=dict(
            gridcolor="rgba(0,200,255,0.08)",
            zerolinecolor="rgba(0,200,255,0.1)",
            tickfont=dict(color="#7b8db8"),
        ),
        yaxis=dict(
            gridcolor="rgba(0,200,255,0.08)",
            zerolinecolor="rgba(0,200,255,0.1)",
            tickfont=dict(color="#7b8db8"),
        ),
    )
    return fig


NEON_COLORS = [
    "#00c8ff", "#a855f7", "#00ffe1", "#6366f1",
    "#38bdf8", "#c084fc", "#34d399", "#818cf8",
]

NEON_SEQ = ["#050a18", "#0a1a40", "#0d2060", "#1a3a8a", "#2563eb",
            "#00a8ff", "#00c8ff", "#00ffe1"]


# ═════════════════════════════════════════════════════════════════════════════
#  CACHED DATA & MODEL LOADERS
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def cached_kpis():
    return Q.get_kpis()

@st.cache_data(ttl=300)
def cached_top_users(limit=10):
    return Q.get_top_users(limit)

@st.cache_data(ttl=300)
def cached_trending_hashtags(limit=15):
    return Q.get_trending_hashtags(limit)

@st.cache_data(ttl=300)
def cached_most_liked(limit=10):
    return Q.get_most_liked_posts(limit)

@st.cache_data(ttl=300)
def cached_most_commented(limit=10):
    return Q.get_most_commented_posts(limit)

@st.cache_data(ttl=300)
def cached_community_engagement():
    return Q.get_community_engagement()

@st.cache_data(ttl=300)
def cached_engagement_trend():
    return Q.get_engagement_trend()

@st.cache_data(ttl=300)
def cached_activity_heatmap():
    return Q.get_activity_heatmap()

@st.cache_data(ttl=300)
def cached_posts_ml():
    return Q.get_all_posts_for_ml()

@st.cache_data(ttl=300)
def cached_user_profiles():
    return Q.get_user_profiles()

@st.cache_resource
def cached_engagement_model():
    return train_engagement_model()

@st.cache_resource
def cached_sentiment_model():
    return train_sentiment_model()

@st.cache_resource
def cached_spam_model():
    return train_spam_model()


# ═════════════════════════════════════════════════════════════════════════════
#  AUTH PAGES
# ═════════════════════════════════════════════════════════════════════════════

def show_login():
    st.markdown("""
    <div class='auth-container'>
        <div class='moon-ring'>🌙</div>
        <div class='brand-title'>Social Verse AI</div>
        <div class='brand-sub'>Intelligence Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown("##### Sign In to Your Account")
        username = st.text_input("Username", placeholder="Enter username", key="login_user")
        password = st.text_input("Password", placeholder="Enter password", type="password", key="login_pass")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 Login", use_container_width=True, key="btn_login"):
                if username and password:
                    user = login_user(username, password)
                    if user:
                        st.session_state.user = user
                        st.session_state.logged_in = True
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password.")
                else:
                    st.warning("Please fill in all fields.")
        with col_btn2:
            if st.button("📝 Sign Up", use_container_width=True, key="btn_go_signup"):
                st.session_state.show_signup = True
                st.rerun()

        st.markdown("""
        <div style='text-align:center; color:#7b8db8; font-size:0.78rem; margin-top:1rem;'>
            Demo credentials: <code style='color:#00c8ff'>admin</code> / <code style='color:#00c8ff'>admin123</code>
        </div>
        """, unsafe_allow_html=True)


def show_signup():
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; margin-bottom:1.5rem;'>
            <div style='font-size:2.5rem;'>🌐</div>
            <div class='brand-title'>Create Account</div>
            <div class='brand-sub'>Join Social Verse AI</div>
        </div>
        """, unsafe_allow_html=True)

        username  = st.text_input("Username *", placeholder="Choose a username", key="su_user")
        email     = st.text_input("Email *",    placeholder="your@email.com",     key="su_email")
        password  = st.text_input("Password *", placeholder="Min 6 characters",   type="password", key="su_pass")
        bio       = st.text_area("Bio",         placeholder="Tell us about yourself (optional)", key="su_bio", height=80)
        interests = st.text_input("Interests",  placeholder="e.g. python, ai, data science", key="su_interests")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ Register", use_container_width=True, key="btn_register"):
                if not all([username, email, password]):
                    st.warning("Username, email, and password are required.")
                elif len(password) < 6:
                    st.warning("Password must be at least 6 characters.")
                else:
                    result = register_user(username, email, password, bio, interests)
                    if result["success"]:
                        st.success("✅ Account created! Please log in.")
                        st.session_state.show_signup = False
                        st.rerun()
                    else:
                        st.error(result["error"])
        with col_b:
            if st.button("← Back to Login", use_container_width=True, key="btn_back"):
                st.session_state.show_signup = False
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        # ── Logo + brand header ──────────────────────────────────────────────
        st.markdown("""
        <div style='text-align:center; padding: 1.2rem 0 1rem;'>
            <div class='sidebar-logo' style='font-size:3rem; margin-bottom:6px;'>🌐</div>
            <div style='font-family:"Space Grotesk",sans-serif; font-size:1.15rem; font-weight:700;
                        background:linear-gradient(135deg,#00c8ff,#a855f7);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text; letter-spacing:0.5px;'>
                Social Verse AI<span class='nav-cursor'></span>
            </div>
            <div style='font-size:0.7rem; color:#5a6e98; letter-spacing:2px;
                        text-transform:uppercase; margin-top:3px;'>
                Intelligence Dashboard
            </div>
        </div>
        <div style='height:1px; background:linear-gradient(90deg,transparent,rgba(0,200,255,0.3),transparent);
                    margin:0 0 1rem;'></div>
        """, unsafe_allow_html=True)

        # ── User info card ───────────────────────────────────────────────────
        user = st.session_state.get("user", {})
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(0,200,255,0.06),rgba(168,85,247,0.04));
                    border:1px solid rgba(0,200,255,0.18);
                    border-radius:12px; padding:0.75rem 1rem; margin-bottom:1.1rem;
                    box-shadow: 0 0 12px rgba(0,200,255,0.05);'>
            <div style='font-size:0.68rem; color:#5a6e98; text-transform:uppercase;
                        letter-spacing:1.2px; margin-bottom:4px;'>Active Session</div>
            <div style='display:flex; align-items:center; gap:8px;'>
                <span class='pulse-dot'></span>
                <span style='font-weight:700; color:#00c8ff; font-size:0.92rem;'>
                    {user.get("username","")}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Section label ────────────────────────────────────────────────────
        st.markdown("""
        <div style='font-size:0.65rem; color:#3d5070; text-transform:uppercase;
                    letter-spacing:2px; padding:0 4px; margin-bottom:6px;'>
            Navigation
        </div>
        """, unsafe_allow_html=True)

        # ── Nav radio ────────────────────────────────────────────────────────
        nav = st.radio(
            "Navigation",
            [
                "🏠 Overview",
                "👥 Users",
                "📝 Posts",
                "🔖 Hashtags",
                "🏘️ Communities",
                "🔍 Search & Analyze",
                "🤖 ML Predictions",
                "💬 Sentiment Analysis",
                "🔗 Recommendations",
            ],
            label_visibility="collapsed",
        )

        # ── Divider ──────────────────────────────────────────────────────────
        st.markdown("""
        <div style='height:1px; background:linear-gradient(90deg,transparent,rgba(0,200,255,0.2),transparent);
                    margin:1rem 0;'></div>
        """, unsafe_allow_html=True)

        # ── Logout button ─────────────────────────────────────────────────────
        if st.button("Logout", use_container_width=True, key="btn_logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()

        # ── Footer ───────────────────────────────────────────────────────────
        st.markdown("""
        <div style='margin-top:1.8rem; text-align:center;'>
            <div style='font-size:0.68rem; color:#2a3a5a; letter-spacing:0.5px;
                        border-top:1px solid rgba(0,200,255,0.06); padding-top:1rem;'>
                AI Social Verse <span style='color:#00c8ff88;'>v1.0</span><br>
                <span style='color:#1e2d4a;'>Intelligence Dashboard &copy; 2025</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    return nav


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

def page_overview():
    st.markdown("""
    <div class='section-header'>
        <span style='font-size:1.8rem;'>🏠</span>
        <h2>Platform Overview</h2>
    </div>
    """, unsafe_allow_html=True)

    kpis = cached_kpis()

    # ── KPI Row ───────────────────────────────────────────────────────────────
    kpi_data = [
        ("👥", "Total Users",         f"{kpis['total_users']:,}"),
        ("📝", "Total Posts",         f"{kpis['total_posts']:,}"),
        ("💬", "Total Comments",      f"{kpis['total_comments']:,}"),
        ("❤️", "Total Likes",         f"{kpis['total_likes']:,}"),
        ("🔁", "Total Shares",        f"{kpis['total_shares']:,}"),
        ("📊", "Engagement Rate",     f"{kpis['engagement_rate']}x"),
        ("🏘️", "Top Community",       kpis['most_active_community'][:16]),
    ]

    cols = st.columns(len(kpi_data))
    for col, (icon, label, val) in zip(cols, kpi_data):
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-icon'>{icon}</div>
                <div class='kpi-value'>{val}</div>
                <div class='kpi-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### 📈 Engagement Trend (Last 365 Days)")
        trend = cached_engagement_trend()
        if not trend.empty:
            # Rolling 7-day average for smoother curve
            trend_r = trend.set_index("date").rolling(7).mean().reset_index()
            fig = go.Figure()
            fill_alpha = {"#00c8ff": "rgba(0,200,255,0.07)", "#a855f7": "rgba(168,85,247,0.07)", "#00ffe1": "rgba(0,255,225,0.07)"}
            for metric, color in [("likes","#00c8ff"), ("comments","#a855f7"), ("shares","#00ffe1")]:
                fig.add_trace(go.Scatter(
                    x=trend_r["date"], y=trend_r[metric],
                    name=metric.capitalize(), mode="lines",
                    line=dict(color=color, width=2),
                    fill="tozeroy",
                    fillcolor=fill_alpha[color],
                ))
            fig = dark_fig(fig, height=320)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🥧 Media Type Distribution")
        posts_df = cached_posts_ml()
        media_counts = posts_df["media_type"].value_counts().reset_index()
        media_counts.columns = ["type", "count"]
        fig = px.pie(
            media_counts, values="count", names="type",
            color_discrete_sequence=NEON_COLORS, hole=0.55,
        )
        fig.update_traces(textfont_color="#e8eeff")
        fig = dark_fig(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)

    # ── Charts Row 2 ──────────────────────────────────────────────────────────
    col3, col4 = st.columns([1, 2])

    with col3:
        st.markdown("#### 🏆 Top 5 Users by Engagement")
        top_u = cached_top_users(5)
        fig = px.bar(
            top_u, x="total_engagement", y="username",
            orientation="h", color="total_engagement",
            color_continuous_scale=[[0, "#1a1f6e"], [0.5, "#00c8ff"], [1, "#a855f7"]],
        )
        fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Engagement")
        fig = dark_fig(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("#### 🗓️ User Activity Heatmap (Day × Hour)")
        heatmap = cached_activity_heatmap()
        fig = px.imshow(
            heatmap, color_continuous_scale=NEON_SEQ,
            labels=dict(x="Hour of Day", y="Day", color="Posts"),
            aspect="auto",
        )
        fig = dark_fig(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: USERS
# ═════════════════════════════════════════════════════════════════════════════

def page_users():
    st.markdown("""
    <div class='section-header'>
        <span style='font-size:1.8rem;'>👥</span>
        <h2>User Analytics</h2>
    </div>
    """, unsafe_allow_html=True)

    top_users = cached_top_users(20)
    profiles  = cached_user_profiles()

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown("#### 🏆 Top Users by Total Engagement")
        fig = px.bar(
            top_users.head(15), x="username", y="total_engagement",
            color="total_engagement",
            color_continuous_scale=[[0,"#0d2060"],[0.5,"#00c8ff"],[1,"#a855f7"]],
            text="total_engagement",
        )
        fig.update_traces(textposition="outside", textfont_color="#e8eeff")
        fig.update_layout(xaxis_tickangle=-35, coloraxis_showscale=False)
        fig = dark_fig(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📊 Engagement Breakdown")
        fig = px.bar(
            top_users.head(10), x="username",
            y=["likes_received", "comments_received", "shares_received"],
            barmode="stack",
            color_discrete_sequence=["#00c8ff", "#a855f7", "#00ffe1"],
        )
        fig.update_layout(xaxis_tickangle=-35, legend_title="")
        fig = dark_fig(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Full User Leaderboard")
    st.dataframe(
        top_users.style.background_gradient(
            subset=["total_engagement"], cmap="Blues"
        ),
        use_container_width=True, hide_index=True,
    )

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 🧑‍💻 Posts Made per User")
        fig = px.histogram(
            profiles, x="posts_made", nbins=20,
            color_discrete_sequence=["#6366f1"],
        )
        fig = dark_fig(fig, height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("#### 💬 Comments Made per User")
        fig = px.histogram(
            profiles, x="comments_made", nbins=20,
            color_discrete_sequence=["#a855f7"],
        )
        fig = dark_fig(fig, height=280)
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: POSTS
# ═════════════════════════════════════════════════════════════════════════════

def page_posts():
    st.markdown("""
    <div class='section-header'>
        <span style='font-size:1.8rem;'>📝</span>
        <h2>Posts Analytics</h2>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔥 Most Liked", "💬 Most Commented", "📊 Engagement Distribution"])

    with tab1:
        liked = cached_most_liked(15)
        col1, col2 = st.columns([1, 1.5])
        with col1:
            fig = px.bar(
                liked, x="likes", y="username", orientation="h",
                color="likes",
                color_continuous_scale=[[0,"#0a1a40"],[1,"#00c8ff"]],
            )
            fig.update_layout(coloraxis_showscale=False)
            fig = dark_fig(fig, height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(liked, use_container_width=True, hide_index=True)

    with tab2:
        commented = cached_most_commented(15)
        col1, col2 = st.columns([1, 1.5])
        with col1:
            fig = px.bar(
                commented, x="comments", y="username", orientation="h",
                color="comments",
                color_continuous_scale=[[0,"#1a0a40"],[1,"#a855f7"]],
            )
            fig.update_layout(coloraxis_showscale=False)
            fig = dark_fig(fig, height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(commented, use_container_width=True, hide_index=True)

    with tab3:
        posts_df = cached_posts_ml()
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                posts_df, x="total_engagement", nbins=30,
                color_discrete_sequence=["#00c8ff"],
                title="Distribution of Total Engagement",
            )
            fig = dark_fig(fig, height=320)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(
                posts_df, x="likes", y="comments", size="shares",
                color="total_engagement",
                color_continuous_scale=[[0,"#050a18"],[0.5,"#00c8ff"],[1,"#a855f7"]],
                title="Likes vs Comments (bubble = shares)",
            )
            fig = dark_fig(fig, height=320)
            st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: HASHTAGS
# ═════════════════════════════════════════════════════════════════════════════

def page_hashtags():
    st.markdown("""
    <div class='section-header'>
        <span style='font-size:1.8rem;'>🔖</span>
        <h2>Hashtag Analytics</h2>
    </div>
    """, unsafe_allow_html=True)

    hashtags = cached_trending_hashtags(20)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("#### 📊 Hashtag Popularity Bar Chart")
        fig = px.bar(
            hashtags, x="hashtag", y="usage_count",
            color="usage_count",
            color_continuous_scale=[[0,"#0d1b4b"],[0.5,"#6366f1"],[1,"#a855f7"]],
            text="usage_count",
        )
        fig.update_traces(textposition="outside", textfont_color="#e8eeff")
        fig.update_layout(xaxis_tickangle=-35, coloraxis_showscale=False)
        fig = dark_fig(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🌐 Trending Hashtag Treemap")
        fig = px.treemap(
            hashtags, path=["hashtag"], values="usage_count",
            color="usage_count",
            color_continuous_scale=[[0,"#0d1b4b"],[0.5,"#00c8ff"],[1,"#00ffe1"]],
        )
        fig.update_traces(textfont_color="#ffffff")
        fig = dark_fig(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🔢 Hashtag Leaderboard")
    st.dataframe(hashtags, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: COMMUNITIES
# ═════════════════════════════════════════════════════════════════════════════

def page_communities():
    st.markdown("""
    <div class='section-header'>
        <span style='font-size:1.8rem;'>🏘️</span>
        <h2>Community Analytics</h2>
    </div>
    """, unsafe_allow_html=True)

    comm_df = cached_community_engagement()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📝 Posts per Community")
        fig = px.bar(
            comm_df, x="community", y="posts",
            color="posts",
            color_continuous_scale=[[0,"#0a1a40"],[1,"#00c8ff"]],
        )
        fig.update_layout(xaxis_tickangle=-30, coloraxis_showscale=False)
        fig = dark_fig(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 👥 Member Count per Community")
        fig = px.bar(
            comm_df, x="community", y="member_count",
            color="member_count",
            color_continuous_scale=[[0,"#1a0a40"],[1,"#a855f7"]],
        )
        fig.update_layout(xaxis_tickangle=-30, coloraxis_showscale=False)
        fig = dark_fig(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📊 Stacked Engagement by Community")
    fig = px.bar(
        comm_df, x="community",
        y=["posts", "likes", "comments", "shares"],
        barmode="stack",
        color_discrete_sequence=["#00c8ff", "#a855f7", "#00ffe1", "#6366f1"],
    )
    fig.update_layout(xaxis_tickangle=-30, legend_title="")
    fig = dark_fig(fig, height=360)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Community Engagement Table")
    st.dataframe(comm_df, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: ML PREDICTIONS
# ═════════════════════════════════════════════════════════════════════════════

def page_ml_predictions():
    st.markdown("""
    <div class='section-header'>
        <span style='font-size:1.8rem;'>🤖</span>
        <h2>ML Predictions — Engagement Forecaster</h2>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔄 Loading RandomForest model..."):
        model, accuracy, feature_cols, report = cached_engagement_model()

    st.markdown(f"""
    <div class='glass-box'>
        <b>Model:</b> RandomForestClassifier &nbsp;|&nbsp;
        <b>Task:</b> Binary Engagement Classification (High / Low) &nbsp;|&nbsp;
        <b>Accuracy:</b> <span class='acc-badge'>🎯 {accuracy}%</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### ✍️ Predict Engagement for a New Post")
        content    = st.text_area("Post Content", placeholder="Write your post here...", height=110, key="pred_content")
        media_type = st.selectbox("Media Type", ["text", "image", "video", "link"], key="pred_media")
        hour       = st.slider("Posting Hour (0–23)", 0, 23, 12, key="pred_hour")

        if st.button("⚡ Predict Engagement", use_container_width=True, key="btn_predict"):
            if content.strip():
                result = predict_engagement(model, content, media_type, hour)
                st.markdown(f"""
                <div class='glass-box' style='text-align:center; padding:1.5rem;'>
                    <div style='font-size:1.8rem; font-weight:700; color:#00c8ff;'>{result['prediction']}</div>
                    <div style='color:#7b8db8; margin-top:6px;'>Confidence: <b style='color:#a855f7'>{result['confidence']}%</b></div>
                    <div style='margin-top:1rem; display:flex; gap:1rem; justify-content:center;'>
                        <div style='background:rgba(0,200,255,0.1); border-radius:8px; padding:8px 18px;'>
                            🔥 High: <b>{result['prob_high']}%</b>
                        </div>
                        <div style='background:rgba(168,85,247,0.1); border-radius:8px; padding:8px 18px;'>
                            📉 Low: <b>{result['prob_low']}%</b>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Please enter some post content.")

    with col2:
        st.markdown("#### 📊 Feature Importances")
        fi = get_feature_importances(model, feature_cols)
        fig = px.bar(
            fi, x="Importance", y="Feature", orientation="h",
            color="Importance",
            color_continuous_scale=[[0,"#0d1b4b"],[0.5,"#00c8ff"],[1,"#a855f7"]],
        )
        fig.update_layout(coloraxis_showscale=False)
        fig = dark_fig(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Classification Report")
    with st.expander("View Detailed Report", expanded=False):
        st.code(report, language="text")

    # Spam Detection sub-section
    st.markdown("---")
    st.markdown("""
    <div class='section-header'>
        <span style='font-size:1.5rem;'>🚫</span>
        <h2>Spam / Fake Post Detector</h2>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔄 Loading Spam Detection model..."):
        spam_vec, spam_model, spam_acc, spam_report, spam_cm = cached_spam_model()

    st.markdown(f"""
    <div class='glass-box'>
        <b>Model:</b> TF-IDF + Logistic Regression &nbsp;|&nbsp;
        <b>Task:</b> Binary Spam Classification &nbsp;|&nbsp;
        <b>Accuracy:</b> <span class='acc-badge'>🎯 {spam_acc}%</span>
    </div>
    """, unsafe_allow_html=True)

    col3, col4 = st.columns([1.2, 1])

    with col3:
        spam_text = st.text_area("Test Post for Spam", placeholder="Paste a post to check...", height=100, key="spam_input")
        if st.button("🔍 Check for Spam", use_container_width=True, key="btn_spam"):
            if spam_text.strip():
                res = predict_spam(spam_vec, spam_model, spam_text)
                color = "#ff4060" if res["is_spam"] else "#00ffe1"
                st.markdown(f"""
                <div class='glass-box' style='text-align:center; border-color:{color}50;'>
                    <div style='font-size:1.6rem; font-weight:700; color:{color};'>{res['label']}</div>
                    <div style='color:#7b8db8; margin-top:6px;'>Confidence: <b style='color:{color}'>{res['confidence']}%</b></div>
                    <div style='margin-top:0.7rem; font-size:0.85rem; color:#7b8db8;'>
                        Spam prob: <b>{res['spam_prob']}%</b> &nbsp;|&nbsp; Legit prob: <b>{res['legit_prob']}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Enter some text to analyze.")

    with col4:
        st.markdown("#### 📊 Spam vs Legit Posts in Dataset")
        posts_df = cached_posts_ml()
        spam_counts = posts_df["is_spam"].map({0: "Legitimate", 1: "Spam"}).value_counts().reset_index()
        spam_counts.columns = ["type", "count"]
        fig = px.pie(
            spam_counts, values="count", names="type",
            color_discrete_sequence=["#00ffe1", "#ff4060"], hole=0.5,
        )
        fig = dark_fig(fig, height=280)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Spam Model Classification Report"):
        st.code(spam_report, language="text")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: SENTIMENT ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def page_sentiment():
    st.markdown("""
    <div class='section-header'>
        <span style='font-size:1.8rem;'>💬</span>
        <h2>Sentiment Analysis</h2>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔄 Loading Sentiment model..."):
        sent_vec, sent_model, sent_acc, label_map, sent_report = cached_sentiment_model()

    st.markdown(f"""
    <div class='glass-box'>
        <b>Model:</b> TF-IDF + Logistic Regression (Multinomial) &nbsp;|&nbsp;
        <b>Task:</b> Positive / Neutral / Negative &nbsp;|&nbsp;
        <b>Accuracy:</b> <span class='acc-badge'>🎯 {sent_acc}%</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### ✍️ Analyze Custom Text")
        custom_text = st.text_area("Enter text to analyze", placeholder="Type a post or comment...", height=120, key="sent_input")
        if st.button("🔍 Analyze Sentiment", use_container_width=True, key="btn_sentiment"):
            if custom_text.strip():
                res = predict_sentiment(sent_vec, sent_model, custom_text)
                colors = {"😠 Negative": "#ff4060", "😐 Neutral": "#7b8db8", "😊 Positive": "#00ffe1"}
                color = colors.get(res["label"], "#00c8ff")
                st.markdown(f"""
                <div class='glass-box' style='text-align:center; border-color:{color}60;'>
                    <div style='font-size:2rem; font-weight:700; color:{color};'>{res['label']}</div>
                    <div style='display:flex; gap:1rem; justify-content:center; margin-top:1rem; flex-wrap:wrap;'>
                        <div style='background:rgba(255,64,96,0.1); border-radius:8px; padding:6px 14px; font-size:0.85rem;'>
                            😠 Neg: <b>{res['negative']}%</b>
                        </div>
                        <div style='background:rgba(123,141,184,0.1); border-radius:8px; padding:6px 14px; font-size:0.85rem;'>
                            😐 Neutral: <b>{res['neutral']}%</b>
                        </div>
                        <div style='background:rgba(0,255,225,0.1); border-radius:8px; padding:6px 14px; font-size:0.85rem;'>
                            😊 Pos: <b>{res['positive']}%</b>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Please enter some text.")

    with col2:
        st.markdown("#### 📊 Overall Post Sentiment Distribution")
        posts_df = cached_posts_ml()
        sent_posts = get_bulk_sentiment(sent_vec, sent_model, posts_df.head(200))
        sent_dist  = sent_posts["sentiment"].value_counts().reset_index()
        sent_dist.columns = ["sentiment", "count"]
        fig = px.pie(
            sent_dist, values="count", names="sentiment",
            color_discrete_sequence=["#ff4060", "#7b8db8", "#00ffe1"], hole=0.5,
        )
        fig.update_traces(textfont_color="#e8eeff")
        fig = dark_fig(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Bulk sentiment bar over time
    st.markdown("#### 📈 Sentiment Trend Over Time")
    posts_df2 = cached_posts_ml().copy()
    posts_df2 = get_bulk_sentiment(sent_vec, sent_model, posts_df2)
    posts_df2["month"] = pd.to_datetime(posts_df2["created_at"]).dt.to_period("M").astype(str)
    monthly = posts_df2.groupby(["month", "sentiment"]).size().reset_index(name="count")
    fig = px.bar(
        monthly, x="month", y="count", color="sentiment", barmode="stack",
        color_discrete_map={"Negative":"#ff4060","Neutral":"#6366f1","Positive":"#00ffe1"},
    )
    fig.update_layout(xaxis_tickangle=-30, legend_title="")
    fig = dark_fig(fig, height=350)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Sentiment Model Classification Report"):
        st.code(sent_report, language="text")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════════════════

def page_recommendations():
    st.markdown("""
    <div class='section-header'>
        <span style='font-size:1.8rem;'>🔗</span>
        <h2>User Recommendation System</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-box'>
        <b>Algorithm:</b> TF-IDF vectorization over user interest + activity profiles,
        ranked by <b>Cosine Similarity</b>
    </div>
    """, unsafe_allow_html=True)

    profiles = cached_user_profiles()
    usernames = sorted(profiles["username"].tolist())

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### 🔎 Find Similar Users")
        selected_user = st.selectbox("Select a user", usernames, key="rec_user")
        top_n = st.slider("Number of recommendations", 3, 10, 5, key="rec_n")

        if st.button("🔗 Get Recommendations", use_container_width=True, key="btn_rec"):
            recs = get_recommendations(selected_user, top_n)
            if recs.empty:
                st.info("No recommendations found.")
            else:
                st.session_state["rec_results"] = recs

        if "rec_results" in st.session_state:
            recs = st.session_state["rec_results"]
            for _, row in recs.iterrows():
                st.markdown(f"""
                <div class='glass-box' style='margin-bottom:0.6rem;'>
                    <div style='font-weight:600; color:#00c8ff;'>👤 {row['username']}</div>
                    <div style='font-size:0.8rem; color:#7b8db8; margin:2px 0;'>
                        Interests: {row.get('interests','—')}
                    </div>
                    <div style='font-size:0.8rem;'>
                        Match: <b style='color:#a855f7'>{row['similarity_score']}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 🗺️ User Similarity Heatmap (Top 15 Users)")
        with st.spinner("Computing similarity matrix..."):
            sim_matrix = get_similar_users_matrix(15)

        fig = px.imshow(
            sim_matrix, color_continuous_scale=NEON_SEQ,
            labels=dict(color="Similarity"),
            zmin=0, zmax=1, aspect="auto",
        )
        fig.update_layout(
            xaxis_tickangle=-35,
            xaxis=dict(tickfont=dict(size=9, color="#7b8db8")),
            yaxis=dict(tickfont=dict(size=9, color="#7b8db8")),
        )
        fig = dark_fig(fig, height=480)
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: SEARCH & ANALYZE
# ═════════════════════════════════════════════════════════════════════════════

def page_search_analyze():
    st.markdown("""
    <div class='section-header'>
        <span style='font-size:1.8rem;'>🔍</span>
        <h2>Search &amp; Analyze</h2>
    </div>
    <div class='glass-box' style='margin-bottom:1.4rem;'>
        Search by <b style='color:#00c8ff;'>Username</b>,
        <b style='color:#a855f7;'>Hashtag</b>, or
        <b style='color:#00ffe1;'>Community</b> to get a complete analytics deep-dive.
    </div>
    """, unsafe_allow_html=True)

    tab_user, tab_hash, tab_comm = st.tabs([
        "👤 User Profile", "#️⃣ Hashtag Drill-down", "🏘️ Community Deep-dive"
    ])

    # ── Tab 1: User ───────────────────────────────────────────────────────────
    with tab_user:
        profiles = cached_user_profiles()
        all_usernames = sorted(profiles["username"].tolist())

        col_inp, col_btn = st.columns([3, 1])
        with col_inp:
            uname = st.selectbox(
                "Select or type a username", all_usernames,
                key="sa_username",
                help="Choose any username to see their full analytics profile"
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            search_u = st.button("Analyze User", key="sa_btn_user", use_container_width=True)

        if search_u or "sa_user_result" in st.session_state:
            profile = Q.search_user_profile(uname)
            if profile is None:
                st.error(f"User '{uname}' not found.")
            else:
                st.session_state["sa_user_result"] = profile
                p = profile

                # Profile card
                status_color = "#00ffe1" if p["is_active"] else "#ff4060"
                status_text  = "Active" if p["is_active"] else "Inactive"
                st.markdown(f"""
                <div class='glass-box' style='display:flex; gap:2rem; flex-wrap:wrap; align-items:flex-start;'>
                    <div style='font-size:3.5rem; filter:drop-shadow(0 0 14px #00c8ff);'>👤</div>
                    <div style='flex:1; min-width:200px;'>
                        <div style='font-family:"Space Grotesk",sans-serif; font-size:1.5rem;
                                    font-weight:700; color:#00c8ff;'>@{p['username']}</div>
                        <div style='color:#7b8db8; font-size:0.82rem; margin:4px 0 8px;'>
                            {p['email']} &nbsp;|&nbsp;
                            Joined: {str(p['joined_at'])[:10]} &nbsp;|&nbsp;
                            <span style='color:{status_color};'>{status_text}</span>
                        </div>
                        <div style='font-size:0.88rem; color:#c8d6f0; margin-bottom:6px;'>
                            <b style='color:#7b8db8;'>Bio:</b> {p['bio']}
                        </div>
                        <div style='font-size:0.85rem; color:#c8d6f0;'>
                            <b style='color:#7b8db8;'>Interests:</b>
                            {'  '.join([f"<span style='background:rgba(0,200,255,0.1);border:1px solid rgba(0,200,255,0.2);border-radius:6px;padding:2px 8px;font-size:0.78rem;color:#00c8ff;'>{t.strip()}</span>" for t in p['interests'].split(',')])}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # KPI mini-cards
                kpi_items = [
                    ("📝", "Posts",        p["posts_count"]),
                    ("❤️", "Likes Recv",   p["likes_received"]),
                    ("💬", "Comments Recv",p["comments_recv"]),
                    ("🔁", "Shares Recv",  p["shares_recv"]),
                    ("📊", "Total Eng.",   p["total_engagement"]),
                    ("👥", "Followers",    p["followers"]),
                    ("➡️", "Following",    p["following"]),
                ]
                cols = st.columns(len(kpi_items))
                for col, (icon, label, val) in zip(cols, kpi_items):
                    with col:
                        st.markdown(f"""
                        <div class='kpi-card'>
                            <div class='kpi-icon'>{icon}</div>
                            <div class='kpi-value' style='font-size:1.5rem;'>{val:,}</div>
                            <div class='kpi-label'>{label}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                col_left, col_right = st.columns([1.4, 1])
                with col_left:
                    st.markdown("#### 📝 Top Posts by Engagement")
                    user_posts = Q.get_user_posts(uname, 12)
                    if user_posts.empty:
                        st.info("No posts found.")
                    else:
                        fig = px.bar(
                            user_posts, x="engagement", y="preview",
                            orientation="h", color="engagement",
                            color_continuous_scale=[[0,"#0a1a40"],[1,"#00c8ff"]],
                            labels={"preview": "", "engagement": "Engagement"},
                        )
                        fig.update_layout(
                            coloraxis_showscale=False,
                            yaxis=dict(tickfont=dict(size=9)),
                        )
                        fig = dark_fig(fig, height=360)
                        st.plotly_chart(fig, use_container_width=True)

                with col_right:
                    st.markdown("#### 📅 Monthly Activity Timeline")
                    timeline = Q.get_user_activity_timeline(uname)
                    if timeline.empty:
                        st.info("No activity data.")
                    else:
                        fig = px.bar(
                            timeline, x="month", y="posts",
                            color="posts",
                            color_continuous_scale=[[0,"#0a1540"],[1,"#a855f7"]],
                        )
                        fig.update_layout(
                            xaxis_tickangle=-35, coloraxis_showscale=False
                        )
                        fig = dark_fig(fig, height=360)
                        st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### 📋 Post Details Table")
                st.dataframe(
                    Q.get_user_posts(uname, 20),
                    use_container_width=True, hide_index=True
                )

    # ── Tab 2: Hashtag ────────────────────────────────────────────────────────
    with tab_hash:
        st.markdown("""
        <div class='glass-box' style='padding:0.8rem 1.2rem; margin-bottom:1rem;'>
            <b style='color:#a855f7;'>Type any hashtag</b> below (with or without #)
            to get a full analytics breakdown — trend, top users, and all posts.
        </div>
        """, unsafe_allow_html=True)

        col_inp2, col_btn2, col_clr2 = st.columns([2.5, 0.9, 0.6])
        with col_inp2:
            tag_input = st.text_input(
                "Enter hashtag",
                placeholder="e.g.  python   or   #machinelearning",
                key="sa_hashtag_text",
                label_visibility="collapsed",
            )
        with col_btn2:
            search_h = st.button("Analyze", key="sa_btn_hash", use_container_width=True)
        with col_clr2:
            if st.button("Clear", key="sa_clr_hash", use_container_width=True):
                st.session_state.pop("sa_hash_result", None)
                st.rerun()

        # ─ Suggestion chips from known trending tags ─────────────────────────
        known_tags = cached_trending_hashtags(30)["hashtag"].tolist()
        chips_html = " ".join(
            f"<span style='background:rgba(168,85,247,0.12); border:1px solid rgba(168,85,247,0.3);"
            f"border-radius:16px; padding:3px 12px; font-size:0.78rem; color:#c084fc;"
            f"margin:2px; display:inline-block; cursor:pointer;'>#{t}</span>"
            for t in known_tags[:20]
        )
        st.markdown(
            f"<div style='margin-bottom:1rem; line-height:2;'>"
            f"<span style='font-size:0.72rem; color:#5a6e98; text-transform:uppercase;"
            f"letter-spacing:1px;'>Trending: </span>{chips_html}</div>",
            unsafe_allow_html=True
        )

        # ─ Run analysis ─────────────────────────────────────────────────────
        raw_tag = tag_input.strip().lstrip("#").lower()

        if search_h and not raw_tag:
            st.warning("Please enter a hashtag name first.")

        elif (search_h and raw_tag) or "sa_hash_result" in st.session_state:
            active_tag = raw_tag if (search_h and raw_tag) else st.session_state.get("sa_hash_result", {}).get("tag", raw_tag)

            # Partial match suggestions when no exact result
            detail = Q.search_hashtag_detail(active_tag)

            if detail["usage_count"] == 0:
                # Try partial match from known tags
                similar = [t for t in known_tags if active_tag in t or t in active_tag]
                st.warning(f"Hashtag '#{active_tag}' not found exactly.")
                if similar:
                    st.markdown(
                        f"<div style='margin-top:0.5rem;'>Did you mean: "
                        + " ".join(
                            f"<b style='color:#a855f7;'>#{s}</b>"
                            for s in similar[:6]
                        ) + "?</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.session_state["sa_hash_result"] = detail

                st.markdown(f"""
                <div class='glass-box' style='display:flex; align-items:center; gap:1.5rem; flex-wrap:wrap;'>
                    <div style='font-size:2rem;'>🔖</div>
                    <div>
                        <div style='font-size:1.5rem; font-weight:700; color:#a855f7;
                                    font-family:"Space Grotesk",sans-serif;'>#{detail['tag']}</div>
                        <div style='color:#7b8db8; font-size:0.82rem; margin-top:2px;'>
                            Found in <b style='color:#00c8ff;'>{detail['usage_count']}</b> posts
                        </div>
                    </div>
                    <span class='acc-badge' style='margin-left:auto;'>{detail['usage_count']} total uses</span>
                </div>
                """, unsafe_allow_html=True)

                col_l, col_r = st.columns([1.2, 1])
                with col_l:
                    st.markdown("#### 📈 Monthly Usage Trend")
                    trend_df = Q.get_hashtag_trend(active_tag)
                    if not trend_df.empty:
                        fig = px.area(
                            trend_df, x="month", y="usage",
                            color_discrete_sequence=["#a855f7"],
                        )
                        fig.update_traces(fill="tozeroy", fillcolor="rgba(168,85,247,0.1)")
                        fig.update_layout(xaxis_tickangle=-35)
                        fig = dark_fig(fig, height=300)
                        st.plotly_chart(fig, use_container_width=True)

                with col_r:
                    st.markdown("#### 🏆 Top Users Using This Hashtag")
                    top_users_tag = Q.get_hashtag_top_users(active_tag, 10)
                    if not top_users_tag.empty:
                        fig = px.bar(
                            top_users_tag, x="post_count", y="username",
                            orientation="h", color="post_count",
                            color_continuous_scale=[[0,"#1a0a40"],[1,"#a855f7"]],
                        )
                        fig.update_layout(coloraxis_showscale=False)
                        fig = dark_fig(fig, height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No user data for this hashtag.")

                st.markdown("#### 📋 Posts Using This Hashtag")
                hash_posts = Q.get_hashtag_posts(active_tag, 25)
                st.dataframe(hash_posts, use_container_width=True, hide_index=True)

    # ── Tab 3: Community ──────────────────────────────────────────────────────
    with tab_comm:
        st.markdown("""
        <div class='glass-box' style='padding:0.8rem 1.2rem; margin-bottom:1rem;'>
            <b style='color:#00ffe1;'>Type any community name</b> (full or partial keyword)
            to explore contributors, posts, and engagement stats.
        </div>
        """, unsafe_allow_html=True)

        # Suggestion chips from all communities
        all_comms = Q.get_community_engagement()["community"].tolist()
        comm_chips = " ".join(
            f"<span style='background:rgba(0,255,225,0.07); border:1px solid rgba(0,255,225,0.22);"
            f"border-radius:16px; padding:3px 12px; font-size:0.78rem; color:#00ffe1;"
            f"margin:2px; display:inline-block;'>{c}</span>"
            for c in all_comms
        )
        st.markdown(
            f"<div style='margin-bottom:1rem; line-height:2;'>"
            f"<span style='font-size:0.72rem; color:#5a6e98; text-transform:uppercase;"
            f"letter-spacing:1px;'>Available: </span>{comm_chips}</div>",
            unsafe_allow_html=True
        )

        col_inp3, col_btn3, col_clr3 = st.columns([2.5, 0.9, 0.6])
        with col_inp3:
            comm_input = st.text_input(
                "Enter community name or keyword",
                placeholder="e.g.  Python   or   Machine Learning   or   AI",
                key="sa_community_text",
                label_visibility="collapsed",
            )
        with col_btn3:
            search_c = st.button("Analyze", key="sa_btn_comm", use_container_width=True)
        with col_clr3:
            if st.button("Clear", key="sa_clr_comm", use_container_width=True):
                st.session_state.pop("sa_comm_result", None)
                st.rerun()

        raw_comm = comm_input.strip()

        if search_c and not raw_comm:
            st.warning("Please enter a community name or keyword.")

        elif (search_c and raw_comm) or "sa_comm_result" in st.session_state:
            active_comm = raw_comm if (search_c and raw_comm) else st.session_state.get("sa_comm_result", {}).get("name", raw_comm)
            detail_c = Q.search_community_detail(active_comm)

            if detail_c is None:
                # Partial match suggestions
                similar_comms = [c for c in all_comms if active_comm.lower() in c.lower()]
                st.warning(f"No community matched '{active_comm}'.")
                if similar_comms:
                    st.markdown(
                        "<div>Similar communities: "
                        + " | ".join(f"<b style='color:#00ffe1;'>{s}</b>" for s in similar_comms)
                        + "</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.info("Try a shorter keyword like 'python', 'AI', 'data', 'cloud', etc.")
            else:
                st.session_state["sa_comm_result"] = detail_c
                d = detail_c

                st.markdown(f"""
                <div class='glass-box' style='display:flex; flex-wrap:wrap; gap:2rem; align-items:flex-start;'>
                    <div style='font-size:3rem;'>🏘️</div>
                    <div style='flex:1; min-width:220px;'>
                        <div style='font-size:1.4rem; font-weight:700; color:#00ffe1;
                                    font-family:"Space Grotesk",sans-serif;'>
                            {d['name']}
                        </div>
                        <div style='color:#7b8db8; font-size:0.82rem; margin:4px 0 8px;'>
                            Created: {str(d['created_at'])[:10]} &nbsp;|&nbsp;
                            Members: <b style='color:#00c8ff;'>{d['member_count']:,}</b>
                        </div>
                        <div style='font-size:0.85rem; color:#c8d6f0;'>{d['description']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                comm_kpis = [
                    ("📝", "Posts",       d["total_posts"]),
                    ("❤️", "Likes",       d["total_likes"]),
                    ("💬", "Comments",    d["total_comments"]),
                    ("🔁", "Shares",      d["total_shares"]),
                    ("📊", "Engagement",  d["total_engagement"]),
                    ("👥", "Members",     d["member_count"]),
                ]
                cols = st.columns(len(comm_kpis))
                for col, (icon, label, val) in zip(cols, comm_kpis):
                    with col:
                        st.markdown(f"""
                        <div class='kpi-card'>
                            <div class='kpi-icon'>{icon}</div>
                            <div class='kpi-value' style='font-size:1.5rem;'>{val:,}</div>
                            <div class='kpi-label'>{label}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                col_cl, col_cr = st.columns(2)
                with col_cl:
                    st.markdown("#### 🏆 Top Contributors")
                    contribs = Q.get_community_top_contributors(d["name"], 10)
                    if not contribs.empty:
                        fig = px.bar(
                            contribs, x="username", y="posts",
                            color="engagement",
                            color_continuous_scale=[[0,"#0a2040"],[1,"#00ffe1"]],
                            text="posts",
                        )
                        fig.update_traces(textposition="outside", textfont_color="#e8eeff")
                        fig.update_layout(xaxis_tickangle=-30, coloraxis_showscale=False)
                        fig = dark_fig(fig, height=320)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No contributor data yet.")

                with col_cr:
                    st.markdown("#### 📊 Engagement Breakdown")
                    if not contribs.empty:
                        fig = px.bar(
                            contribs.head(8), x="username",
                            y=["likes_received", "comments_received"],
                            barmode="stack",
                            color_discrete_sequence=["#00c8ff", "#a855f7"],
                        )
                        fig.update_layout(xaxis_tickangle=-30, legend_title="")
                        fig = dark_fig(fig, height=320)
                        st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### 📋 Top Posts in This Community")
                comm_posts = Q.get_community_posts(d["name"], 15)
                st.dataframe(comm_posts, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # Initialize session state keys
    if "logged_in"   not in st.session_state: st.session_state.logged_in   = False
    if "user"        not in st.session_state: st.session_state.user        = None
    if "show_signup" not in st.session_state: st.session_state.show_signup = False

    if not st.session_state.logged_in:
        if st.session_state.show_signup:
            show_signup()
        else:
            show_login()
        return

    # ── Authenticated layout ──
    nav = render_sidebar()

    if   nav == "🏠 Overview":           page_overview()
    elif nav == "👥 Users":              page_users()
    elif nav == "📝 Posts":              page_posts()
    elif nav == "🔖 Hashtags":           page_hashtags()
    elif nav == "🏘️ Communities":        page_communities()
    elif nav == "🔍 Search & Analyze":   page_search_analyze()
    elif nav == "🤖 ML Predictions":     page_ml_predictions()
    elif nav == "💬 Sentiment Analysis": page_sentiment()
    elif nav == "🔗 Recommendations":    page_recommendations()


if __name__ == "__main__":
    main()
