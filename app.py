import os
import io
import csv
import sqlite3
from datetime import datetime
from collections import Counter

from flask import Flask, render_template, request, redirect, url_for, send_file
from transformers import pipeline
import pandas as pd
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR

app = Flask(__name__)
DB_NAME = "reviews.db"

# ---------------------------------------------------------
# 1. DATABASE SETUP (SQLite - Built right into Python!)
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            score REAL NOT NULL,
            emotion TEXT NOT NULL,
            aspects TEXT,
            source TEXT,
            created_at TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------
# 2. LOAD HUGGING FACE AI MODELS (Downloads once on first run)
# ---------------------------------------------------------
print("Loading AI Models... Please wait a moment on first startup.")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)
emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS (Aspect Extraction & Word Clouds)
# ---------------------------------------------------------
ASPECT_KEYWORDS = {
    "Quality": ["quality", "build", "material", "durable", "broken", "cheap", "sturdy", "design"],
    "Price": ["price", "cost", "expensive", "cheap", "value", "money", "worth", "affordable"],
    "Delivery": ["delivery", "shipping", "arrived", "late", "fast", "packaging", "package", "delay"],
    "Performance": ["battery", "speed", "fast", "slow", "screen", "camera", "sound", "work", "working"],
    "Support": ["service", "support", "customer", "refund", "return", "staff", "help"]
}

def extract_aspects(text, overall_sentiment):
    text_lower = text.lower()
    found = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(word in text_lower for word in keywords):
            found.append(f"{aspect}:{overall_sentiment}")
    return ", ".join(found) if found else "General:" + overall_sentiment

def analyze_and_save(text, source="Manual"):
    clean_text = str(text).strip()
    if len(clean_text) < 3:
        return
    # Limit text length to 512 chars so transformer model runs fast
    truncated = clean_text[:512]

    sent_res = sentiment_analyzer(truncated)[0]
    label_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
    sentiment = label_map.get(sent_res["label"].lower(), sent_res["label"].capitalize())
    score = round(sent_res["score"] * 100, 1)

    emo_res = emotion_analyzer(truncated)[0]
    emotion = emo_res["label"].capitalize()

    aspects = extract_aspects(clean_text, sentiment)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO reviews (text, sentiment, score, emotion, aspects, source, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (clean_text, sentiment, score, emotion, aspects, source, created_at))
    conn.commit()
    conn.close()

def generate_wordclouds(rows):
    pos_text = " ".join([r[1] for r in rows if r[2] == "Positive"])
    neg_text = " ".join([r[1] for r in rows if r[2] == "Negative"])

    os.makedirs("static", exist_ok=True)
    pos_path = os.path.join("static", "pos_cloud.png")
    neg_path = os.path.join("static", "neg_cloud.png")

    has_pos, has_neg = False, False

    if len(pos_text.split()) >= 3:
        wc_pos = WordCloud(width=500, height=260, background_color="#111827", colormap="cool").generate(pos_text)
        wc_pos.to_file(pos_path)
        has_pos = True

    if len(neg_text.split()) >= 3:
        wc_neg = WordCloud(width=500, height=260, background_color="#111827", colormap="autumn").generate(neg_text)
        wc_neg.to_file(neg_path)
        has_neg = True

    return has_pos, has_neg

# ---------------------------------------------------------
# 4. WEB ROUTES (Dashboard, Single Input, CSV, YouTube, Export)
# ---------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reviews ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    total = len(rows)
    pos_count = sum(1 for r in rows if r[2] == "Positive")
    neg_count = sum(1 for r in rows if r[2] == "Negative")
    neu_count = sum(1 for r in rows if r[2] == "Neutral")

    # Emotion breakdown
    emotions = Counter([r[4] for r in rows])

    # Aspect breakdown
    aspect_stats = {}
    for r in rows:
        if r[5]:
            for item in r[5].split(", "):
                if ":" in item:
                    asp, sent = item.split(":")
                    if asp not in aspect_stats:
                        aspect_stats[asp] = {"Positive": 0, "Negative": 0, "Neutral": 0}
                    aspect_stats[asp][sent] = aspect_stats[asp].get(sent, 0) + 1

    # Key Takeaways Summary
    takeaways = []
    if total > 0:
        pos_pct = round((pos_count / total) * 100)
        neg_pct = round((neg_count / total) * 100)
        takeaways.append(f"{pos_pct}% of all analyzed feedback is Positive, while {neg_pct}% is Negative.")
        if emotions:
            top_emo = emotions.most_common(1)[0]
            takeaways.append(f"Dominant customer emotion detected is '{top_emo[0]}' ({top_emo[1]} mentions).")
        # Find most complained aspect
        worst_asp = None
        max_neg = 0
        for asp, counts in aspect_stats.items():
            if asp != "General" and counts["Negative"] > max_neg:
                max_neg = counts["Negative"]
                worst_asp = asp
        if worst_asp:
            takeaways.append(f"Top pain point: '{worst_asp}' received {max_neg} negative review(s).")

    has_pos_cloud, has_neg_cloud = generate_wordclouds(rows)

    return render_template(
        "index.html",
        reviews=rows[:30],
        total=total,
        pos_count=pos_count,
        neg_count=neg_count,
        neu_count=neu_count,
        emotion_labels=list(emotions.keys()),
        emotion_values=list(emotions.values()),
        aspect_stats=aspect_stats,
        takeaways=takeaways,
        has_pos_cloud=has_pos_cloud,
        has_neg_cloud=has_neg_cloud
    )

@app.route("/analyze", methods=["POST"])
def analyze():
    review_text = request.form.get("review_text", "").strip()
    if review_text:
        analyze_and_save(review_text, source="Single Input")
    return redirect(url_for("index"))

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    file = request.files.get("csv_file")
    if file and file.filename.endswith(".csv"):
        df = pd.read_csv(file)
        # Pick the first text column (or column named 'review'/'text'/'comment')
        target_col = df.columns[0]
        for col in df.columns:
            if col.lower() in ["review", "reviews", "text", "comment", "comments", "feedback"]:
                target_col = col
                break
        # Analyze up to 30 rows at a time for fast response
        for text in df[target_col].dropna().head(30):
            analyze_and_save(str(text), source="CSV Upload")
    return redirect(url_for("index"))

@app.route("/scrape_youtube", methods=["POST"])
def scrape_youtube():
    yt_url = request.form.get("youtube_url", "").strip()
    if yt_url:
        try:
            downloader = YoutubeCommentDownloader()
            comments = downloader.get_comments_from_url(yt_url, sort_by=SORT_BY_POPULAR)
            count = 0
            for comment in comments:
                if count >= 15:  # Pull top 15 comments for quick showcase
                    break
                analyze_and_save(comment["text"], source="YouTube")
                count += 1
        except Exception as e:
            print("YouTube scraping error:", e)
    return redirect(url_for("index"))

@app.route("/export_csv")
def export_csv():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, text, sentiment, score, emotion, aspects, source, created_at FROM reviews")
    rows = cursor.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Review Text", "Sentiment", "Confidence (%)", "Emotion", "Aspects", "Source", "Timestamp"])
    writer.writerows(rows)

    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="sentiment_report.csv")

@app.route("/clear", methods=["POST"])
def clear_data():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM reviews")
    conn.commit()
    conn.close()
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)