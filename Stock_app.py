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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Stock Intelligence (Tech+Sentiment+ML)", layout="wide")
st.title("🔬 Stock Intelligence — Indicators, Sentiment, Geo & ML (Optional)")

# ---------------------------
# Helper: dynamic NSE constituent fetch (same as previous dynamic app)
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

@st.cache_data(ttl=24*60*60)
def fetch_index_constituents(which: str) -> pd.DataFrame:
    urls = INDEX_URLS[which]
    last_err = None
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(BytesIO(r.content))
            cols = {c.strip().lower(): c for c in df.columns}
            sym_col = cols.get("symbol") or cols.get("ticker") or list(df.columns)[0]
            name_col = cols.get("company name") or cols.get("companyname") or (list(df.columns)[1] if len(df.columns)>1 else list(df.columns)[0])
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
# Sidebar: user controls
# ---------------------------
st.sidebar.header("Universe & Performance Controls")
indices = st.sidebar.multiselect("Indices to include", ["MIDCAP100","SMALLCAP100"], default=["MIDCAP100","SMALLCAP100"])
limit_universe = st.sidebar.number_input("Max tickers (0=all)", min_value=0, value=0, step=10)
calc_sentiment = st.sidebar.checkbox("Compute company & geo sentiment (slower)", value=True)
headlines_per_company = st.sidebar.slider("Headlines per company", 1, 6, 4)
top_n_chart = st.sidebar.slider("Top N for bar chart", 5, 50, 20)

st.sidebar.markdown("### Scoring weights (sum not required, normalized internally)")
w_momentum = st.sidebar.slider("Momentum weight", 0.0, 1.0, 0.4, 0.05)
w_news = st.sidebar.slider("Company news weight", 0.0, 1.0, 0.3, 0.05)
w_geo = st.sidebar.slider("Geopolitics weight", 0.0, 1.0, 0.3, 0.05)

st.sidebar.markdown("### ML (optional)")
enable_ml = st.sidebar.checkbox("Enable/train ML model (may be slow)", value=False)
ml_days_history = st.sidebar.number_input("Days history for features", min_value=60, max_value=750, value=250, step=10)

# ---------------------------
# Geopolitics topics and mapping (example mapping — extend as desired)
# ---------------------------
geo_topics = ["Oil prices","US Fed rates","India-China relations","Global recession"]
geo_mapping = {
    "Oil prices": ["ONGC.NS","BPCL.NS","RELIANCE.NS","PETRONET.NS"],
    "US Fed rates": [],  # banks mapped by sector below
    "India-China relations": [], 
    "Global recession": []
}
# simple sector mapping fallback: banks list from fetched constituents later

# ---------------------------
# Sentiment helpers
# ---------------------------
@st.cache_data(ttl=6*60*60)
def company_sentiment(company_name: str, max_heads: int = 4) -> float:
    try:
        g = GoogleNews(lang="en", period="1d")
        g.search(company_name + " India")
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
def topic_sentiment(topic: str, max_heads: int = 6) -> float:
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

def polarity_label(p):
    if p > 0.1: return "Positive"
    if p < -0.1: return "Negative"
    return "Neutral"

# ---------------------------
# Fetch universe
# ---------------------------
st.info("Fetching index constituents (may take a few seconds)...")
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
    st.error(f"Could not fetch constituents: {e}")
    st.stop()

universe = sorted(list(dict.fromkeys(universe)))
if limit_universe and limit_universe > 0:
    universe = universe[:limit_universe]

st.write(f"Universe size: *{len(universe)}*")

# If geo_mapping banks empty, fill with bank tickers if present
bank_candidates = [t for t in universe if "BANK" in t or "HDFC" in t or "ICICI" in t]
if not geo_mapping["US Fed rates"]:
    geo_mapping["US Fed rates"] = bank_candidates[:20]

# ---------------------------
# Download historical price data (vectorized)
# ---------------------------
days_hist = max(ml_days_history, 250)
st.subheader("Downloading price history...")
with st.spinner("Downloading with yfinance (this may take ~30-90s for many tickers)..."):
    price_df = yf.download(universe, period=f"{days_hist+5}d", interval="1d", group_by="ticker", threads=True, auto_adjust=False, progress=False)

# helper to extract series safely
def get_close_series(ticker):
    try:
        s = price_df[ticker]['Close'].dropna()
        return s
    except Exception:
        try:
            # yfinance sometimes returns single-column DataFrame for 1 ticker
            df = yf.download(ticker, period=f"{days_hist+5}d", interval="1d", progress=False)
            return df['Close'].dropna()
        except Exception:
            return pd.Series(dtype=float)

# ---------------------------
# Technical indicators functions
# ---------------------------
def pct_return(series, days):
    if len(series) < days+1: return np.nan
    return ((series.iloc[-1] - series.iloc[-1-days]) / series.iloc[-1-days]) * 100.0

def volatility_30d(series):
    returns = series.pct_change().dropna()
    return returns.tail(30).std() * 100.0 if len(returns) >= 30 else np.nan

def moving_average(series, n):
    if len(series) < n: return np.nan
    return series.rolling(window=n).mean().iloc[-1]

def rsi(series, n=14):
    # standard RSI calculation
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=(n-1), adjust=False).mean()
    ma_down = down.ewm(com=(n-1), adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.iloc[-1] if not rsi_series.empty else np.nan

# ---------------------------
# Compute features for universe
# ---------------------------
st.subheader("Computing indicators & (optional) sentiment...")
rows = []
# precompute topic sentiments if enabled
topic_scores = {}
if calc_sentiment:
    for topic in geo_topics:
        topic_scores[topic] = topic_sentiment(topic, max_heads=6)
else:
    for topic in geo_topics:
        topic_scores[topic] = 0.0

for t in universe:
    s = get_close_series(t)
    if s.empty:
        rows.append({
            "Ticker": t, "Company": name_map.get(t,t.replace(".NS","")), "5d": np.nan, "30d": np.nan,
            "Vol30": np.nan, "RSI14": np.nan, "MA20": np.nan, "MA50": np.nan, "MA200": np.nan,
            "SentScore": 0.0, "GeoScore": 0.0
        })
        continue
    five = pct_return(s, 5)
    thirty = pct_return(s, 30)
    vol30 = volatility_30d(s)
    rsi14 = rsi(s, 14)
    ma20 = moving_average(s, 20)
    ma50 = moving_average(s, 50)
    ma200 = moving_average(s, 200)

    # company sentiment
    comp_name = name_map.get(t, t.replace(".NS",""))
    comp_sent = company_sentiment(comp_name, max_heads=headlines_per_company) if calc_sentiment else 0.0

    # geopolitics score: average of mapped topics that affect this ticker
    geo_vals = []
    for topic, mapped in geo_mapping.items():
        if t in mapped or any(key in comp_name.upper() for key in ["OIL","PETRO","ENERGY"]) and topic=="Oil prices":
            geo_vals.append(topic_scores.get(topic,0.0))
        # fallback: if banks and US Fed topic
        if "BANK" in t and topic == "US Fed rates":
            geo_vals.append(topic_scores.get(topic,0.0))
    geo_score = float(np.mean(geo_vals)) if geo_vals else 0.0

    rows.append({
        "Ticker": t,
        "Company": comp_name,
        "5d": five, "30d": thirty, "Vol30": vol30,
        "RSI14": rsi14, "MA20": ma20, "MA50": ma50, "MA200": ma200,
        "SentScore": round(comp_sent,4), "GeoScore": round(geo_score,4)
    })

features_df = pd.DataFrame(rows)

# ---------------------------
# Build composite momentum score from technicals
# Normalize features across universe (z-score) then compose momentum
# ---------------------------
tech_df = features_df.copy()
# Select numeric columns for momentum
tech_cols = ["5d","30d","Vol30","RSI14"]
for c in tech_cols:
    tech_df[c+"_z"] = (tech_df[c] - tech_df[c].mean()) / (tech_df[c].std() + 1e-9)

# Momentum = (z(30d return) + z(5d return)*0.7 + z(RSI)*0.3 - z(Volatility)*0.5) normalized
tech_df["MomentumScore"] = (
    (tech_df["30d_z"].fillna(0) * 0.6) +
    (tech_df["5d_z"].fillna(0) * 0.4) +
    (tech_df["RSI14_z"].fillna(0) * 0.1) -
    (tech_df["Vol30_z"].fillna(0) * 0.2)
)
# normalize momentum to -1..1 via tanh
tech_df["MomentumScore"] = np.tanh(tech_df["MomentumScore"].fillna(0))

# Merge Momentum + Sentiment to compute Final Score using user weights
features_df = features_df.merge(tech_df[["Ticker","MomentumScore"]], on="Ticker", how="left")

# normalize weights
total_w = w_momentum + w_news + w_geo
if total_w == 0:
    total_w = 1.0
wn = w_news/total_w; wm = w_momentum/total_w; wg = w_geo/total_w

features_df["FinalScore"] = (wm * features_df["MomentumScore"]) + (wn * features_df["SentScore"]) + (wg * features_df["GeoScore"])

# attach latest % change for sorting / bar chart
def last_pct_change(ticker):
    s = get_close_series(ticker)
    if len(s) < 2: return np.nan
    return round(((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2]) * 100.0,2)
features_df["LastDay%"] = features_df["Ticker"].apply(last_pct_change)

# sort by FinalScore
out_df = features_df.sort_values(by="FinalScore", ascending=False).reset_index(drop=True)
st.subheader("Results — sorted by Final Score")
st.dataframe(out_df[["Ticker","Company","FinalScore","MomentumScore","SentScore","GeoScore","LastDay%"]].fillna("N/A"), use_container_width=True)

# ---------------------------
# Charts
# ---------------------------
st.subheader("Charts")
topn = top_n_chart
chart_df = out_df.dropna(subset=["LastDay%"]).head(topn)
if not chart_df.empty:
    fig, ax = plt.subplots(figsize=(12,4))
    ax.bar(chart_df["Ticker"], chart_df["LastDay%"])
    ax.set_ylabel("% change (last day)")
    ax.set_title(f"Top {len(chart_df)} by Final Score — Last Day % change")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# Final score histogram
fig2, ax2 = plt.subplots(figsize=(6,3))
ax2.hist(out_df["FinalScore"].dropna(), bins=30)
ax2.set_title("Distribution of Final Scores")
st.pyplot(fig2)

# ---------------------------
# Optional ML: train a logistic regression to predict next-day up/down
# ---------------------------
if enable_ml:
    st.subheader("🔁 Train ML model (Logistic Regression)")
    st.info("Model trains on historical features to predict next-day Up (>0%) probability. This can take a few minutes.")
    if st.button("Start training ML model"):
        # Prepare historical dataset
        X_rows = []
        y_rows = []
        tickers_for_ml = universe  # use current universe
        for ticker in tickers_for_ml:
            s = get_close_series(ticker)
            if len(s) < 60: continue
            dfp = pd.DataFrame({"Close": s})
            dfp["Ret1"] = dfp["Close"].pct_change()
            dfp["Ret5"] = dfp["Close"].pct_change(5)
            dfp["Ret20"] = dfp["Close"].pct_change(20)
            dfp["MA5"] = dfp["Close"].rolling(5).mean()
            dfp["MA20"] = dfp["Close"].rolling(20).mean()
            dfp["Vol20"] = dfp["Ret1"].rolling(20).std()
            dfp = dfp.dropna()
            # Build features & target on rolling window
            for i in range(30, len(dfp)-1):
                feat = {
                    "Ret5": dfp["Ret5"].iloc[i],
                    "Ret20": dfp["Ret20"].iloc[i],
                    "MA5_MA20": (dfp["MA5"].iloc[i] - dfp["MA20"].iloc[i]) / (dfp["MA20"].iloc[i]+1e-9),
                    "Vol20": dfp["Vol20"].iloc[i]
                }
                # target: next day return > 0
                target = 1 if dfp["Close"].iloc[i+1] > dfp["Close"].iloc[i] else 0
                X_rows.append(feat)
                y_rows.append(target)
        if not X_rows:
            st.warning("Not enough historical data to train ML model.")
        else:
            X = pd.DataFrame(X_rows)
            y = np.array(y_rows)
            # simple scaling + logistic regression
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X.fillna(0))
            Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=200)
            model.fit(Xtr, ytr)
            score = model.score(Xte, yte)
            st.success(f"Model trained. Test accuracy ≈ {round(score*100,2)}%")
            # predict current probability for universe using latest features
            preds = []
            for ticker in universe:
                s = get_close_series(ticker)
                if len(s) < 30:
                    preds.append(np.nan)
                    continue
                dfp = pd.DataFrame({"Close": s})
                dfp["Ret1"] = dfp["Close"].pct_change()
                dfp["Ret5"] = dfp["Close"].pct_change(5)
                dfp["Ret20"] = dfp["Close"].pct_change(20)
                dfp["MA5"] = dfp["Close"].rolling(5).mean()
                dfp["MA20"] = dfp["Close"].rolling(20).mean()
                dfp["Vol20"] = dfp["Ret1"].rolling(20).std()
                dfp = dfp.dropna()
                if dfp.empty:
                    preds.append(np.nan)
                    continue
                i = -1
                feat = {
                    "Ret5": dfp["Ret5"].iloc[i],
                    "Ret20": dfp["Ret20"].iloc[i],
                    "MA5_MA20": (dfp["MA5"].iloc[i] - dfp["MA20"].iloc[i]) / (dfp["MA20"].iloc[i]+1e-9),
                    "Vol20": dfp["Vol20"].iloc[i]
                }
                Xcur = scaler.transform(pd.DataFrame([feat]).fillna(0))
                p = model.predict_proba(Xcur)[0,1]
                preds.append(round(float(p),4))
            out_df["ML_Up_Prob"] = preds
            st.dataframe(out_df[["Ticker","Company","FinalScore","ML_Up_Prob"]].head(50))

# ---------------------------
# Download results
# ---------------------------
st.subheader("Download results")
buf = BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    out_df.to_excel(writer, index=False, sheet_name="Scores")
st.download_button("Download Excel (full)", data=buf.getvalue(), file_name=f"stock_scores_{date.today()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Tip: disable company/geo sentiment to speed up runs while testing. ML is optional and can be slow for large universes.")
