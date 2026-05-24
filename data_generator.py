"""
data_generator.py — Generate realistic sample social media data for the dashboard.
Called once during DB initialization if the database is empty.
"""

import random
import sqlite3
from datetime import datetime, timedelta

# ── Seed constants ────────────────────────────────────────────────────────────
random.seed(42)

USERNAMES = [
    "alex_nova", "priya_codes", "john_dev", "sara_ml", "kai_data",
    "luna_ai", "ravi_tech", "emma_nlp", "omar_cloud", "zoe_stats",
    "leo_graph", "nina_ui", "arjun_sys", "clara_ds", "felix_net",
    "maya_vis", "rahul_api", "sophie_sec", "tom_mlops", "jade_db",
    "ivan_cv", "aisha_nlp", "ben_rust", "diana_k8s", "carlos_spark",
    "yuki_dl", "fatima_bi", "noah_re", "anna_gis", "mike_devops",
]

BIOS = [
    "Data science enthusiast | Python lover",
    "AI researcher & open-source contributor",
    "Full-stack developer | Coffee addict",
    "Machine learning engineer at heart",
    "Building the future with neural networks",
    "NLP nerd & competitive programmer",
    "Cloud architect | Kubernetes fan",
    "Data visualization storyteller",
    "Backend dev | Open-source advocate",
    "Deep learning researcher",
]

INTERESTS = [
    "machine learning,python,data science",
    "nlp,text mining,linguistics",
    "web development,react,nodejs",
    "cloud computing,aws,devops",
    "computer vision,opencv,deep learning",
    "data visualization,tableau,d3js",
    "cybersecurity,ethical hacking,linux",
    "blockchain,web3,smart contracts",
    "robotics,iot,embedded systems",
    "statistics,r,probability",
]

COMMUNITIES = [
    ("AI & Machine Learning", "Discuss AI trends, papers, and projects"),
    ("Python Developers", "Python tips, tricks, and best practices"),
    ("Data Science Hub", "Data analysis, visualization, and insights"),
    ("NLP Research", "Natural Language Processing discussions"),
    ("Cloud & DevOps", "Cloud architecture and CI/CD pipelines"),
    ("Computer Vision", "Image processing and vision models"),
    ("Web Development", "Frontend, backend, and full-stack talks"),
    ("Cybersecurity", "Ethical hacking and security news"),
    ("Open Source", "Open-source projects and contributions"),
    ("Student Coders", "Coding challenges and learning resources"),
]

POST_TEMPLATES = [
    "Just finished implementing {topic} using Python. The results are incredible! #python #{tag}",
    "New blog post: Understanding {topic} from scratch. Check it out! #{tag} #learning",
    "Hot take: {topic} is going to change everything in tech by 2026. Do you agree? #{tag}",
    "Working on a {topic} project. Facing issues with performance — any tips? #{tag} #help",
    "Excited to share my {topic} model achieved 94% accuracy! #{tag} #machinelearning",
    "Tutorial: How to use {topic} efficiently in production environments. #{tag} #tutorial",
    "Just published my research paper on {topic}. Years of work finally out! #{tag} #research",
    "The future of {topic} looks promising. Here's why I think so... #{tag} #futuretech",
    "Can we talk about how underrated {topic} is? More people should know this! #{tag}",
    "Daily reminder: {topic} is a skill worth mastering in 2025. #{tag} #career",
    "Live coding session on {topic} tonight at 8 PM. Join me! #{tag} #livestream",
    "Found a critical bug in a popular {topic} library. Here's the fix! #{tag} #opensource",
    "Comparing {topic} vs alternatives — pros and cons. #{tag} #comparison",
    "My weekend project: built a mini {topic} app in 48 hours. #{tag} #weekend",
    "Poll: What's your preferred approach for {topic}? Vote below! #{tag} #poll",
]

COMMENT_TEMPLATES = [
    "This is really insightful, thanks for sharing!",
    "I had the same issue — great solution!",
    "Could you share the source code?",
    "Amazing work! Keep it up!",
    "I disagree on the performance point, but interesting nonetheless.",
    "This saved me hours of debugging. Thank you!",
    "Great tutorial! Subscribed for more.",
    "Have you tried using {alt} instead? Might be faster.",
    "Love this topic! Following for more updates.",
    "Can you do a follow-up on the advanced concepts?",
    "Brilliant explanation — very easy to follow.",
    "I've been looking for this exact solution!",
    "Mind sharing your benchmarks?",
    "Interesting perspective — never thought of it this way.",
    "This is underrated content. More people should see it!",
]

TOPICS = [
    ("transformers", "nlp"), ("random forest", "ml"), ("kubernetes", "devops"),
    ("neural networks", "deeplearning"), ("data pipelines", "dataengineering"),
    ("cosine similarity", "ml"), ("sentiment analysis", "nlp"),
    ("feature engineering", "datascience"), ("docker containers", "devops"),
    ("graph neural networks", "gnn"), ("time series forecasting", "ml"),
    ("web scraping", "python"), ("model deployment", "mlops"),
    ("transfer learning", "deeplearning"), ("sql optimization", "database"),
    ("pandas tricks", "python"), ("object detection", "computervision"),
    ("text classification", "nlp"), ("reinforcement learning", "ai"),
    ("API design", "webdev"),
]

ALL_HASHTAGS = [
    "python", "ml", "ai", "datascience", "nlp", "deeplearning", "devops",
    "cloudcomputing", "opensource", "tech", "coding", "programming",
    "machinelearning", "bigdata", "analytics", "visualization", "research",
    "tutorial", "career", "innovation",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def rand_ts(days_back: int = 365) -> str:
    dt = datetime.now() - timedelta(
        days=random.randint(0, days_back),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ── Main generator ────────────────────────────────────────────────────────────

def generate_all_data(conn: sqlite3.Connection):
    """Insert all sample data into an empty database."""
    cur = conn.cursor()

    # 1. Communities
    community_ids = []
    for name, desc in COMMUNITIES:
        cur.execute(
            "INSERT INTO communities (name, description, created_at, member_count) VALUES (?,?,?,?)",
            (name, desc, rand_ts(500), random.randint(50, 5000))
        )
        community_ids.append(cur.lastrowid)

    # 2. Users (30 sample + 1 demo admin)
    user_ids = []
    for i, uname in enumerate(USERNAMES):
        cur.execute(
            "INSERT INTO users (username, email, password, bio, interests, joined_at) VALUES (?,?,?,?,?,?)",
            (
                uname,
                f"{uname}@example.com",
                # demo password stored hashed: "password123"
                "ef92b778bafe771207860ded37ba8b1e36b4ca9fa2c7f4f3a1bfa5ae2de54e64",
                random.choice(BIOS),
                random.choice(INTERESTS),
                rand_ts(500),
            )
        )
        user_ids.append(cur.lastrowid)

    # Admin / demo user
    import hashlib
    cur.execute(
        "INSERT INTO users (username, email, password, bio, interests, joined_at) VALUES (?,?,?,?,?,?)",
        (
            "admin",
            "admin@socialverse.ai",
            hashlib.sha256("admin123".encode()).hexdigest(),
            "Platform administrator",
            "administration,analytics,ml",
            rand_ts(600),
        )
    )
    user_ids.append(cur.lastrowid)

    # 3. Followers
    for uid in user_ids:
        targets = random.sample([x for x in user_ids if x != uid], k=random.randint(3, 12))
        for tid in targets:
            try:
                cur.execute(
                    "INSERT INTO followers (follower_id, following_id, followed_at) VALUES (?,?,?)",
                    (uid, tid, rand_ts(400))
                )
            except sqlite3.IntegrityError:
                pass

    # 4. Posts
    post_ids = []
    post_contents = []
    for _ in range(300):
        uid = random.choice(user_ids)
        cid = random.choice(community_ids)
        topic, tag = random.choice(TOPICS)
        tmpl = random.choice(POST_TEMPLATES)
        content = tmpl.format(topic=topic, tag=tag)
        is_spam = 1 if random.random() < 0.08 else 0
        if is_spam:
            spam_templates = [
                "WIN A FREE iPhone 15 Pro! Click link in bio NOW! Limited time!! #giveaway #free",
                "Make $5000/day from home!! No experience needed! DM me #money #opportunity",
                "BUY FOLLOWERS CHEAP!! 10K followers for $5 only! DM NOW #followers #viral",
            ]
            content = random.choice(spam_templates)
        ts = rand_ts(365)
        cur.execute(
            "INSERT INTO posts (user_id, community_id, content, media_type, created_at, is_spam) VALUES (?,?,?,?,?,?)",
            (uid, cid, content, random.choice(["text", "image", "video", "link"]), ts, is_spam)
        )
        pid = cur.lastrowid
        post_ids.append(pid)
        post_contents.append((pid, content, uid))

    # 5. Hashtags
    for pid, content, _ in post_contents:
        tags_in_content = [w[1:] for w in content.split() if w.startswith("#") and len(w) > 1]
        extra = random.sample(ALL_HASHTAGS, k=random.randint(0, 2))
        for tag in set(tags_in_content + extra):
            cur.execute("INSERT INTO hashtags (tag, post_id) VALUES (?,?)", (tag.lower(), pid))

    # 6. Likes
    for pid in post_ids:
        likers = random.sample(user_ids, k=random.randint(0, min(20, len(user_ids))))
        for uid in likers:
            try:
                cur.execute(
                    "INSERT INTO likes (post_id, user_id, created_at) VALUES (?,?,?)",
                    (pid, uid, rand_ts(300))
                )
            except sqlite3.IntegrityError:
                pass

    # 7. Shares
    for pid in post_ids:
        sharers = random.sample(user_ids, k=random.randint(0, min(8, len(user_ids))))
        for uid in sharers:
            cur.execute(
                "INSERT INTO shares (post_id, user_id, created_at) VALUES (?,?,?)",
                (pid, uid, rand_ts(300))
            )

    # 8. Comments
    alts = ["PyTorch", "TensorFlow", "scikit-learn", "FastAPI", "Spark"]
    for pid, content, author_uid in post_contents:
        commenters = random.sample(user_ids, k=random.randint(0, min(10, len(user_ids))))
        for uid in commenters:
            tmpl = random.choice(COMMENT_TEMPLATES)
            comment = tmpl.format(alt=random.choice(alts)) if "{alt}" in tmpl else tmpl
            cur.execute(
                "INSERT INTO comments (post_id, user_id, content, created_at) VALUES (?,?,?,?)",
                (pid, uid, comment, rand_ts(280))
            )

    conn.commit()
    print("[data_generator] Sample data inserted successfully.")
