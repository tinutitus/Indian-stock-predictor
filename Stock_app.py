# stock_app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import BytesIO
from GoogleNews import GoogleNews
from textblob import TextBlob
import matplotlib.pyplot as plt
from datetime import date, timedelta

st.set_page_config(page_title="Stock Signals — Indicators + Sentiment + Geo", layout="wide")
st.title("🔎 Stock Signals — Technicals + News + Geopolitics (Midcap + Smallcap)")

# ---------------------------
# Constants & index CSV sources
# ---------------------------
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.niftyindices.com/"}
INDEX_URLS = {
    "MIDCAP100": [
        "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
        "https://www1.nseindia.com/content/indices/ind_niftymidcap100list.csv",
    ],
    "SMALLCAP100": [
        "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap100list.csv",
        "https://www1.nseindia.com/content/indices/ind_niftysmallcap100list.csv",
    ],
}

# ---------------------------
# Helpers: fetch constituents dynamically
# ---------------------------
@st.cache_data(ttl=24*60*60)
def fetch_index_constituents(which: str) -> pd.DataFrame:
    urls = INDEX_URLS.get(which, [])
    last_err = None
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(BytesIO(r.content))
            cols = {c.strip().lower(): c for c in df.columns}
            sym_col = cols.get("symbol") or cols.get("ticker") or list(df.columns)[0]
            name_col = cols.get("company name") or cols.get("companyname") or (list(df.columns)[1] if len(df.columns) > 1 else sym_col)
            out = df[[sym_col, name_col]].copy()
            out.columns = ["Symbol", "Company Name"]
            out["Symbol"] = out["Symbol"].astype(str).str.strip().str.upper()
            out["Ticker"] = out["Symbol"].apply(lambda s: f"{s}.NS")
            out = out.dropna(subset=["Symbol"]).drop_duplicates(subset=["Symbol"]).reset_index(drop=True)
            return out
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to fetch {which} list. Last error: {last_err}")

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Universe & options")
indices = st.sidebar.multiselect("Include indices", ["MIDCAP100", "SMALLCAP100"], default=["MIDCAP100","SMALLCAP100"])
limit_universe = st.sidebar.number_input("Max tickers (0 = all)", min_value=0, value=0, step=10)
calc_sentiment = st.sidebar.checkbox("Compute news sentiment (slower)", value=True)
headlines_per_company = st.sidebar.slider("Headlines per company", min_value=1, max_value=6, value=3)
top_n_chart = st.sidebar.slider("Top N for bar chart", 5, 30, 15)

st.sidebar.markdown("### Scoring weights (normalized internally)")
w_momentum = st.sidebar.slider("Momentum weight", 0.0, 1.0, 0.4, 0.05)
w_news = st.sidebar.slider("Company news weight", 0.0, 1.0, 0.3, 0.05)
w_geo = st.sidebar.slider("Geopolitics weight", 0.0, 1.0, 0.3, 0.05)

st.sidebar.markdown("---")
st.sidebar.write("Tip: turn off sentiment to speed up when testing.")

# ---------------------------
# Geopolitical topics and mapping (affects geo score)
# ---------------------------
geo_topics = ["Oil prices", "US Fed rates", "India-China relations", "Global inflation"]
# mapping for specific tickers that should be affected by topics (extendable)
geo_mapping = {
    "Oil prices": [],  # will detect oil/petro names heuristically
    "US Fed rates": [],  # banks (auto-detected)
    "India-China relations": [],  # export/IT/manufacturing (heuristic)
    "Global inflation": []
}

# ---------------------------
# Sentiment helpers (cached)
# ---------------------------
@st.cache_data(ttl=6*60*60)
def fetch_company_sentiment(company_name: str, max_heads: int = 3) -> float:
    try:
        g = GoogleNews(lang="en", period="1d")
        g.search(f"{company_name} India")
        news = g.result()[:max_heads]
        if not news:
            return 0.0
        scores = []
        for n in news:
            title = n.get("title","")
            if title:
                scores.append(TextBlob(title).sentiment.polarity)
        return float(sum(scores)/len(scores)) if scores else 0.0
    except Exception:
        return 0.0

@st.cache_data(ttl=6*60*60)
def fetch_topic_sentiment(topic: str, max_heads: int = 6) -> float:
    try:
        g = GoogleNews(lang="en", period="1d")
        g.search(topic)
        news = g.result()[:max_heads]
        if not news:
            return 0.0
        scores = []
        for n in news:
            title = n.get("title","")
            if title:
                scores.append(TextBlob(title).sentiment.polarity)
        return float(sum(scores)/len(scores)) if scores else 0.0
    except Exception:
        return 0.0

def polarity_label(p: float) -> str:
    if p > 0.1: return "Positive"
    if p < -0.1: return "Negative"
    return "Neutral"

# ---------------------------
# Build Universe
# ---------------------------
st.info("Fetching index constituents...")
universe = []
name_map = {}
try:
    if "MIDCAP100" in indices:
        dfm = fetch_index_constituents("MIDCAP100")
        universe += dfm["Ticker"].tolist()
        name_map.update(dict(zip(dfm["Ticker"], dfm["Company Name"])))
    if "SMALLCAP100" in indices:
        dfs = fetch_index_constituents("SMALLCAP100")
        universe += dfs["Ticker"].tolist()
        name_map.update(dict(zip(dfs["Ticker"], dfs["Company Name"])))
except Exception as e:
    st.error(f"Error fetching constituents: {e}")
    st.stop()

universe = sorted(list(dict.fromkeys(universe)))
if limit_universe and limit_universe > 0:
    universe = universe[:limit_universe]

st.write(f"Universe size: *{len(universe)}*")

if not universe:
    st.warning("No tickers selected.")
    st.stop()

# ---------------------------
# Download price history (vectorized) and helpers
# ---------------------------
days_needed = 250  # to compute MA200 etc
st.subheader("Downloading price history (may take some time for large universe)...")
with st.spinner("Downloading prices from Yahoo Finance..."):
    price_data = yf.download(universe, period=f"{days_needed+5}d", interval="1d", group_by="ticker", threads=True, progress=False)

def get_close_series(ticker):
    try:
        s = price_data[ticker]["Close"].dropna()
        return s
    except Exception:
        # fallback single ticker
        try:
            df = yf.download(ticker, period=f"{days_needed+5}d", interval="1d", progress=False)
            return df["Close"].dropna()
        except Exception:
            return pd.Series(dtype=float)

# ---------------------------
# Technical indicator functions
# ---------------------------
def pct_return(series, days):
    if len(series) < days+1:
        return np.nan
    return ((series.iloc[-1] - series.iloc[-1-days]) / series.iloc[-1-days]) * 100.0

def vol_30d(series):
    r = series.pct_change().dropna()
    return r.tail(30).std() * 100.0 if len(r) >= 30 else np.nan

def ma(series, n):
    if len(series) < n: return np.nan
    return series.rolling(window=n).mean().iloc[-1]

def rsi(series, n=14):
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=(n-1), adjust=False).mean()
    ma_down = down.ewm(com=(n-1), adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.iloc[-1] if not rsi_series.empty else np.nan

# ---------------------------
# Precompute topic sentiments (global)
# ---------------------------
st.subheader("Fetching geopolitical topic sentiment...")
topic_scores = {}
if calc_sentiment:
    for t in geo_topics:
        topic_scores[t] = fetch_topic_sentiment(t, max_heads=6)
else:
    for t in geo_topics:
        topic_scores[t] = 0.0

# ---------------------------
# Compute features for each ticker
# ---------------------------
st.subheader("Computing indicators & scores...")
rows = []
for tk in universe:
    s = get_close_series(tk)
    comp_name = name_map.get(tk, tk.replace(".NS",""))
    if s.empty:
        rows.append({
            "Ticker": tk, "Company": comp_name, "5d": np.nan, "30d": np.nan, "Vol30": np.nan,
            "RSI14": np.nan, "MA20": np.nan, "MA50": np.nan, "MA200": np.nan,
            "SentScore": 0.0, "GeoScore": 0.0, "LastDay%": np.nan
        })
        continue

    five = pct_return(s, 5)
    thirty = pct_return(s, 30)
    vol30 = vol_30d(s)
    rsi14 = rsi(s, 14)
    ma20 = ma(s, 20)
    ma50 = ma(s, 50)
    ma200 = ma(s, 200)
    lastday = np.nan
    if len(s) >= 2:
        lastday = round(((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2]) * 100.0, 2)

    # Sentiment
    comp_sent = fetch_company_sentiment(comp_name, max_heads=headlines_per_company) if calc_sentiment else 0.0

    # Geopolitics: heuristics mapping
    geo_vals = []
    # oil-related check
    if any(k in comp_name.upper() for k in ["OIL","PETRO","GAS","ENERGY","LNG"]):
        geo_vals.append(topic_scores.get("Oil prices", 0.0))
    # bank-related
    if any(k in tk for k in ["BANK", "FIN", "NBFC", "HDFC", "ICICI", "PNB", "IDFC"]):
        geo_vals.append(topic_scores.get("US Fed rates", 0.0))
    # global themes default
    if not geo_vals:
        # include a small influence of global inflation/topic
        geo_vals.append(topic_scores.get("Global inflation", 0.0))
    geo_score = float(np.mean(geo_vals)) if geo_vals else 0.0

    rows.append({
        "Ticker": tk,
        "Company": comp_name,
        "5d": five,
        "30d": thirty,
        "Vol30": vol30,
        "RSI14": rsi14,
        "MA20": ma20,
        "MA50": ma50,
        "MA200": ma200,
        "SentScore": round(comp_sent,4),
        "GeoScore": round(geo_score,4),
        "LastDay%": lastday
    })

df = pd.DataFrame(rows)

# ---------------------------
# Momentum composition from technicals
# z-score normalize technical inputs then combine
# ---------------------------
tech_cols = ["5d","30d","Vol30","RSI14"]
for c in tech_cols:
    df[c+"_z"] = (df[c] - df[c].mean()) / (df[c].std() + 1e-9)

# momentum formula (tunable)
df["Momentum"] = (
    (df["30d_z"].fillna(0) * 0.6) +
    (df["5d_z"].fillna(0) * 0.4) +
    (df["RSI14_z"].fillna(0) * 0.1) -
    (df["Vol30_z"].fillna(0) * 0.2)
)
df["Momentum"] = np.tanh(df["Momentum"].fillna(0))  # squash to [-1,1]

# ---------------------------
# Final score
# ---------------------------
total_w = w_momentum + w_news + w_geo
if total_w == 0:
    total_w = 1.0
w_news_n = w_news / total_w
w_mom_n = w_momentum / total_w
w_geo_n = w_geo / total_w

df["FinalScore"] = (w_mom_n * df["Momentum"]) + (w_news_n * df["SentScore"]) + (w_geo_n * df["GeoScore"])

# add rank and sorting
df = df.sort_values(by="FinalScore", ascending=False).reset_index(drop=True)

# display main table
st.subheader("Results — sorted by Final Score")
visible_cols = ["Ticker","Company","FinalScore","Momentum","SentScore","GeoScore","LastDay%"]
st.dataframe(df[visible_cols].fillna("N/A"), use_container_width=True)

# ---------------------------
# Charts: top N by FinalScore (LastDay% bar) & score distribution
# ---------------------------
topn = min(top_n_chart, len(df))
chart_df = df.dropna(subset=["LastDay%"]).head(topn)

if not chart_df.empty:
    st.subheader(f"Top {len(chart_df)} by Final Score — Last Day % Change")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(chart_df["Ticker"], chart_df["LastDay%"])
    ax.axhline(0, color="red", linestyle="--")
    ax.set_ylabel("% change (last day)")
    ax.set_xlabel("Ticker")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

st.subheader("Final Score distribution")
fig2, ax2 = plt.subplots(figsize=(6,3))
ax2.hist(df["FinalScore"].dropna(), bins=30)
ax2.set_xlabel("Final Score")
st.pyplot(fig2)

# ---------------------------
# Download Excel
# ---------------------------
st.subheader("Download full data")
out = BytesIO()
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Scores")
st.download_button("Download Excel (Full)", data=out.getvalue(),
                   file_name=f"stock_scores_{date.today()}.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Note: Company & topic sentiment can slow the run for large universes. Use 'Max tickers' to test quickly.")
