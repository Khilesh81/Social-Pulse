# 🌐 AI Social Verse Intelligence Dashboard

> **An advanced AI-powered social media analytics dashboard** built with Python, Streamlit, SQLite, Pandas, Scikit-learn, NLP, and Machine Learning — designed for college final-year presentations.

---

## 📸 Project Overview

The **AI Social Verse Intelligence Dashboard** is a full-featured analytics platform that simulates a real social media environment. It provides deep insights into user behavior, post performance, community growth, and content quality — all powered by machine learning models with a stunning dark neon UI.

---

## ✨ Features

### 🔐 Authentication
- Login & Signup pages with dark neon theme
- SHA-256 hashed password storage
- Session-based login using `st.session_state`
- Persistent user profiles in SQLite

### 🏠 Overview Dashboard
- 7 real-time KPI cards (Users, Posts, Likes, Shares, Engagement Rate, etc.)
- 7-day rolling engagement trend chart
- Media type distribution pie chart
- User activity heatmap (Day × Hour)
- Top 5 users bar chart

### 👥 User Analytics
- Leaderboard with total engagement scores
- Stacked bar chart of likes / comments / shares per user
- Activity distribution histograms

### 📝 Posts Analytics
- Most liked and most commented post rankings
- Engagement distribution histogram
- Likes vs Comments scatter bubble chart

### 🔖 Hashtag Analytics
- Top 20 trending hashtags (bar + treemap)
- Full hashtag leaderboard table

### 🏘️ Community Analytics
- Post count, member count, and stacked engagement per community
- Full engagement breakdown table

### 🤖 ML Predictions
- **Engagement Predictor**: Enter post content → get High/Low engagement prediction with confidence
- Feature importance chart for RandomForest model
- **Spam Detector**: Check any text for spam/fake content
- Spam vs Legit distribution in dataset

### 💬 Sentiment Analysis
- Analyze any custom text for Positive / Neutral / Negative sentiment
- Sentiment distribution pie chart across all posts
- Monthly sentiment trend stacked bar chart

### 🔗 Recommendations
- Enter any username → get top-N most similar users by interest + engagement
- Full 15×15 user similarity heatmap

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit, Custom CSS (Glassmorphism, Neon) |
| **Charts** | Plotly Express & Graph Objects |
| **Backend** | Python 3.10+ |
| **Database** | SQLite3 (via Python built-in) |
| **Data Wrangling** | Pandas, NumPy |
| **ML Framework** | Scikit-learn |
| **NLP** | TF-IDF Vectorizer |
| **Auth** | hashlib (SHA-256), st.session_state |

---

## 🚀 How to Run

### 1. Clone / Download the project

```bash
# Navigate to project folder
cd "kp social serves project"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Open in browser

Streamlit will open at: `http://localhost:8501`

### 5. Login

Use the demo credentials:
- **Username:** `admin`
- **Password:** `admin123`

Or create a new account via the **Sign Up** page.

---

## 🗄️ Database Tables

| Table | Description |
|---|---|
| `users` | User accounts with hashed passwords and interests |
| `communities` | Social groups with member counts |
| `posts` | User posts linked to communities, with spam flag |
| `comments` | Comments on posts |
| `likes` | Like records (user × post, unique constraint) |
| `shares` | Share records (user × post) |
| `hashtags` | Hashtag-to-post mapping |
| `followers` | Follower-following relationships |

---

## 🤖 ML Models Used

### 1. Engagement Prediction
- **Algorithm:** `RandomForestClassifier` (150 trees, max_depth=8)
- **Features:** content length, word count, hashtag presence, question/exclamation marks, link presence, posting hour, media type
- **Target:** Binary — High (≥ median engagement) vs Low

### 2. Sentiment Analysis
- **Algorithm:** `TF-IDF` (5000 features, bigrams) + `LogisticRegression` (multinomial)
- **Classes:** Positive / Neutral / Negative
- **Training data:** Auto-labeled posts + comments using keyword heuristics

### 3. Spam Detection
- **Algorithm:** `TF-IDF` (3000 features, bigrams) + `LogisticRegression` (balanced)
- **Classes:** Legitimate vs Spam
- **Training data:** Posts with `is_spam` ground-truth flag from database

### 4. User Recommendation
- **Algorithm:** `TF-IDF` over interest + activity profile → `cosine_similarity`
- **Input:** Username → finds top-N users with closest profile vectors

---

## 📁 Project Structure

```
kp social serves project/
├── app.py              # Main Streamlit application (all pages, CSS, auth)
├── database.py         # SQLite init, table creation, authentication
├── data_generator.py   # Realistic sample data generation (seeds DB)
├── queries.py          # All SQL analytics queries → Pandas DataFrames
├── ml_model.py         # RandomForest engagement prediction
├── sentiment_model.py  # TF-IDF + LogReg sentiment analysis
├── spam_detector.py    # TF-IDF + LogReg spam detection
├── recommendation.py   # Cosine similarity user recommendations
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── social_verse.db     # SQLite database (auto-created on first run)
```

---

## 🔮 Future Improvements

1. **Deep Learning Sentiment** — Replace Logistic Regression with BERT/DistilBERT for higher accuracy
2. **Real-time Data** — Connect to Twitter/Reddit APIs for live analytics
3. **Collaborative Filtering** — Matrix factorization (SVD) for smarter recommendations
4. **Post Scheduling** — ML-driven optimal time prediction for posting
5. **Graph Analytics** — Community detection using NetworkX on follower graphs
6. **Dashboard Alerts** — Email/SMS alerts for spam spikes or engagement drops
7. **Admin Controls** — Moderation panel to manage spam posts and users
8. **Multi-language Support** — Multilingual sentiment analysis using langdetect + translation APIs
9. **Export Reports** — PDF/Excel report generation from dashboard views
10. **Docker Deployment** — Containerized deployment for cloud hosting

---

## 👨‍💻 Author

Developed as a **Final Year College Project** demonstrating the integration of:
- Web application development (Streamlit)
- Relational databases (SQLite)
- Data analysis (Pandas)
- Machine Learning (Scikit-learn)
- Natural Language Processing (TF-IDF)
- Modern UI/UX design (Glassmorphism + Neon Dark Theme)

---

> *"Turning raw social data into actionable intelligence."*
