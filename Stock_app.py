import streamlit as st
import yfinance as yf
from GoogleNews import GoogleNews
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

# -----------------------------
# Company Lists (Mid + Small Cap)
# -----------------------------
mid_cap_companies = [
    "Adani Power", "Max Healthcare", "Persistent Systems", "HDFC AMC", "Voltas",
    "IDFC First Bank", "Crompton Greaves", "Zydus Lifesciences", "JSW Energy", "Tata Chemicals"
]

small_cap_companies = [
    "Balaji Amines", "Fine Organic", "KEI Industries", "Aarti Drugs", "Deepak Fertilisers",
    "Ruchi Soya", "Birla Corp", "Borosil Renewables", "Triveni Turbine", "Navin Fluorine"
]

companies = mid_cap_companies + small_cap_companies

# -----------------------------
# Geopolitical Topics
# -----------------------------
geo_topics = ["India China relations", "Oil prices", "US Fed interest rates", "India Elections", "Russia Ukraine war"]

# -----------------------------
# Helper Functions
# -----------------------------
def fetch_sentiment(query):
    """Fetch recent news and return sentiment"""
    googlenews = GoogleNews(period="1d")
    googlenews.search(query)
    news = googlenews.result()
    headlines = [item['title'] for item in news]

    sentiment_scores = []
    for hl in headlines:
        blob = TextBlob(hl)
        sentiment_scores.append(blob.sentiment.polarity)

    if not sentiment_scores:
        return "Neutral", headlines

    avg = sum(sentiment_scores) / len(sentiment_scores)
    if avg > 0.05:
        return "Positive", headlines
    elif avg < -0.05:
        return "Negative", headlines
    else:
        return "Neutral", headlines

def sentiment_score(sentiment):
    if sentiment == "Positive":
        return 1
    elif sentiment == "Negative":
        return -1
    else:
        return 0

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Indian Stock Predictor", layout="wide")
st.title("📈 Indian Mid & Small Cap Stock Predictor")
st.write("AI-based daily prediction combining **company news** + **geopolitics**")

results = []
today = dt.date.today()
yesterday = today - dt.timedelta(days=1)

# Fetch geopolitical sentiments once
geo_sentiments = {}
for topic in geo_topics:
    sentiment, _ = fetch_sentiment(topic)
    geo_sentiments[topic] = sentiment

# Process each company
for name in companies:
    company_sentiment, headlines = fetch_sentiment(name + " India")
    company_score = sentiment_score(company_sentiment)

    # Simple mapping example: Oil affects Energy, US Fed affects Banks
    geo_score = 0
    if "Energy" in name or "Power" in name:
        geo_score = sentiment_score(geo_sentiments.get("Oil prices", "Neutral"))
    elif "Bank" in name:
        geo_score = sentiment_score(geo_sentiments.get("US Fed interest rates", "Neutral"))

    # Weighted score (70% company + 30% geo)
    final_score = (0.7 * company_score) + (0.3 * geo_score)

    if final_score > 0.1:
        prediction = "Up"
    elif final_score < -0.1:
        prediction = "Down"
    else:
        prediction = "Flat"

    # Fetch stock data
    ticker_name = yf.Ticker(name + ".NS")
    data = ticker_name.history(period="2d")
    if len(data) >= 2:
        yesterday_close = data['Close'][-2]
        today_close = data['Close'][-1]
        pct_change = ((today_close - yesterday_close) / yesterday_close) * 100
    else:
        pct_change = 0

    results.append({
        "Company": name,
        "Company Sentiment": company_sentiment,
        "Geo Sentiment": geo_score,
        "Final Score": round(final_score, 2),
        "Prediction": prediction,
        "% Change": round(pct_change, 2)
    })

# Create dataframe
df = pd.DataFrame(results)
df = df.sort_values(by="Final Score", ascending=False)
st.dataframe(df, use_container_width=True)

# Bar chart: % Change
st.subheader("📊 Predicted Score vs % Change")
plt.figure(figsize=(10,5))
plt.bar(df["Company"], df["% Change"], color="skyblue")
plt.xticks(rotation=45, ha="right")
plt.axhline(0, color="red", linestyle="--")
plt.ylabel("% Change")
st.pyplot(plt)

# Save to Excel
df.to_excel("daily_predictions.xlsx", index=False)
st.success("✅ Predictions saved to daily_predictions.xlsx")
