# Stock_app.py
"""
Hybrid Ensemble Stock Predictor
- Technical indicators (momentum, RSI, volatility, MAs)
- Company news sentiment + geopolitical topic sentiment
- ML engine: RandomForest predicting probability of >= target% move in 1 month
- FinalScore = weighted blend of Momentum, Sentiment, Geo, ML_Prob
- Shortlist: Price <= 500 AND FinalScore >= threshold
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import BytesIO
from GoogleNews import GoogleNews
from textblob import TextBlob
import matplotlib.pyplot as plt
from datetime import date
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Hybrid Ensemble Stock Predictor", layout="wide")
st.title("🔷 Hybrid Ensemble — Technicals + Sentiment + Geopolitics + ML (1M)")

# ---------------------------
# Sidebar controls (user)
# ---------------------------
st.sidebar.header("Universe & Data")
include_nifty50 = st.sidebar.checkbox("Include NIFTY 50", value=True)
include_midcap = st.sidebar.checkbox("Include NIFTY Midcap 100", value=True)
include_smallcap = st.sidebar.checkbox("Include NIFTY Smallcap 100", value=True)
limit_universe = st.sidebar.number_input("Max tickers (0 = all)", min_value=0, value=0, step=10)

st.sidebar.header("Sentiment / Speed")
calc_sentiment = st.sidebar.checkbox("Compute company & topic sentiment (slower)", value=True)
headlines_per_company = st.sidebar.slider("Headlines per company", 1, 6, 3)

st.sidebar.header("Scoring weights (sum normalized internally)")
w_momentum = st.sidebar.slider("Momentum weight", 0.0, 1.0, 0.4, 0.05)
w_sentiment = st.sidebar.slider("Sentiment weight", 0.0, 1.0, 0.2, 0.05)
w_geo = st.sidebar.slider("Geopolitics weight", 0.0, 1.0, 0.2, 0.05)
w_ml = st.sidebar.slider("ML weight", 0.0, 1.0, 0.2, 0.05)

st.sidebar.header("ML & Shortlist")
enable_ml = st.sidebar.checkbox("Enable ML engine", value=True)
target_pct = st.sidebar.number_input("Target return in 1 month (%)", value=5.0, step=0.5)
prob_threshold = st.sidebar.slider("ML shortlist probability threshold", 0.30, 0.95, 0.65, 0.05)
finalscore_threshold = st.sidebar.slider("Final Score shortlist threshold", 0.1, 0.95, 0.6, 0.05)
max_tickers_for_ml = st.sidebar.number_input("Max tickers for ML (0=all)", min_value=0, value=0, step=10)

st.sidebar.write("Tip: turn off sentiment or limit tickers to speed up testing.")

# ---------------------------
# Universe (simple dynamic fetch or fallback lists)
# ---------------------------
# We will try to fetch official CSV lists from niftyindices; fallback to sample lists if fetch fails.
INDEX_URLS = {
    "NIFTY50": [
        "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
        "https://www1.nseindia.com/content/indices/ind_nifty50list.csv"
    ],
    "MIDCAP100": [
        "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
        "https://www1.nseindia.com/content/indices/ind_niftymidcap100list.csv"
    ],
    "SMALLCAP100": [
        "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap100list.csv",
        "https://www1.nseindia.com/content/indices/ind_niftysmallcap100list.csv"
    ],
}

HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.niftyindices.com/"}

@st.cache_data(ttl=24*60*60)
def fetch_index_tickers(which):
    urls = INDEX_URLS.get(which, [])
    last_err = None
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(BytesIO(r.content))
            cols = {c.strip().lower(): c for c in df.columns}
            sym_col = cols.get("symbol") or cols.get("ticker") or list(df.columns)[0]
            name_col = cols.get("company name") or cols.get("companyname") or (list(df.columns)[1] if len(df.columns)>1 else sym_col)
            out = df[[sym_col, name_col]].copy()
            out.columns = ["Symbol", "Company Name"]
            out["Symbol"] = out["Symbol"].astype(str).str.strip().str.upper()
            out["Ticker"] = out["Symbol"].apply(lambda s: f"{s}.NS")
            out = out.dropna(subset=["Symbol"]).drop_duplicates(subset=["Symbol"]).reset_index(drop=True)
            return out
        except Exception as e:
            last_err = e
            continue
    # fallback: return empty df
    return pd.DataFrame(columns=["Symbol","Company Name","Ticker"])

universe = []
name_map = {}

try:
    if include_nifty50:
        dfn = fetch_index_tickers("NIFTY50")
        if not dfn.empty:
            universe += dfn["Ticker"].tolist()
            name_map.update(dict(zip(dfn["Ticker"], dfn["Company Name"])))
    if include_midcap:
        dfm = fetch_index_tickers("MIDCAP100")
        if not dfm.empty:
            universe += dfm["Ticker"].tolist()
            name_map.update(dict(zip(dfm["Ticker"], dfm["Company Name"])))
    if include_smallcap:
        dfs = fetch_index_tickers("SMALLCAP100")
        if not dfs.empty:
            universe += dfs["Ticker"].tolist()
            name_map.update(dict(zip(dfs["Ticker"], dfs["Company Name"])))
except Exception:
    # if network fetch fails, provide a small fallback list so app still runs
    fallback = ["RELIANCE.NS","INFY.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","MARUTI.NS","TATAMOTORS.NS","ONGC.NS","BPCL.NS"]
    universe += fallback
    for t in fallback:
        name_map[t] = t.replace(".NS","")

# unique & optional limit
universe = sorted(list(dict.fromkeys(universe)))
if limit_universe and limit_universe > 0:
    universe = universe[:limit_universe]

st.write(f"Universe size: **{len(universe)}**")

if len(universe) == 0:
    st.error("No tickers available. Enable at least one index in the sidebar.")
    st.stop()

# ---------------------------
# Download historical prices (vectorized)
# ---------------------------
days_needed = 300
st.subheader("Downloading prices (may take time for large universe)...")
with st.spinner("Downloading price history..."):
    price_data = yf.download(universe, period=f"{days_needed+5}d", interval="1d", group_by="ticker", threads=True, progress=False)

def get_close_series(ticker):
    try:
        s = price_data[ticker]["Close"].dropna()
        return s
    except Exception:
        try:
            df = yf.download(ticker, period=f"{days_needed+5}d", interval="1d", progress=False)
            return df["Close"].dropna()
        except Exception:
            return pd.Series(dtype=float)

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
            t = n.get("title","")
            if t:
                scores.append(TextBlob(t).sentiment.polarity)
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
            t = n.get("title","")
            if t:
                scores.append(TextBlob(t).sentiment.polarity)
        return float(sum(scores)/len(scores)) if scores else 0.0
    except Exception:
        return 0.0

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
    if len(series) < n:
        return np.nan
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
geo_topics = ["Oil prices","US Fed rates","India-China relations","Global inflation"]
topic_scores = {}
if calc_sentiment:
    for t in geo_topics:
        topic_scores[t] = fetch_topic_sentiment(t, max_heads=6)
else:
    for t in geo_topics:
        topic_scores[t] = 0.0

# ---------------------------
# Compute features per ticker
# ---------------------------
st.subheader("Computing indicators & signals...")
rows = []
for t in universe:
    s = get_close_series(t)
    comp_name = name_map.get(t, t.replace(".NS",""))
    if s.empty:
        rows.append({
            "Ticker": t, "Company": comp_name, "Price": np.nan,
            "5d": np.nan, "30d": np.nan, "Vol30": np.nan, "RSI14": np.nan,
            "MA20": np.nan, "MA50": np.nan, "MA200": np.nan,
            "SentScore": 0.0, "GeoScore": 0.0
        })
        continue

    price = float(s.iloc[-1])
    five = pct_return(s, 5)
    thirty = pct_return(s, 30)
    vol30 = vol_30d(s)
    rsi14 = rsi(s, 14)
    ma20 = ma(s, 20)
    ma50 = ma(s, 50)
    ma200 = ma(s, 200)

    comp_sent = fetch_company_sentiment(comp_name, max_heads=headlines_per_company) if calc_sentiment else 0.0

    # geo heuristics
    geo_vals = []
    if any(k in comp_name.upper() for k in ["OIL","PETRO","GAS","ENERGY","LNG","PETRONET","ONGC","BPCL"]):
        geo_vals.append(topic_scores.get("Oil prices", 0.0))
    if any(k in t for k in ["BANK", "HDFC", "ICICI", "AXIS", "KOTAK", "PNB", "YES"]):
        geo_vals.append(topic_scores.get("US Fed rates", 0.0))
    if any(k in comp_name.upper() for k in ["EXPORT","AUTOMOTIVE","AUTO","TATA","MARUTI","MAHINDRA"]):
        geo_vals.append(topic_scores.get("India-China relations", 0.0))
    if not geo_vals:
        geo_vals.append(topic_scores.get("Global inflation", 0.0))
    geo_score = float(np.mean(geo_vals)) if geo_vals else 0.0

    rows.append({
        "Ticker": t,
        "Company": comp_name,
        "Price": price,
        "5d": five,
        "30d": thirty,
        "Vol30": vol30,
        "RSI14": rsi14,
        "MA20": ma20,
        "MA50": ma50,
        "MA200": ma200,
        "SentScore": round(comp_sent,4),
        "GeoScore": round(geo_score,4)
    })

df = pd.DataFrame(rows)

# ---------------------------
# Momentum score (normalize & combine)
# ---------------------------
tech_cols = ["5d","30d","Vol30","RSI14"]
for c in tech_cols:
    df[c + "_z"] = (df[c] - df[c].mean()) / (df[c].std() + 1e-9)

df["Momentum"] = (
    (df["30d_z"].fillna(0) * 0.6) +
    (df["5d_z"].fillna(0) * 0.4) +
    (df["RSI14_z"].fillna(0) * 0.1) -
    (df["Vol30_z"].fillna(0) * 0.2)
)
df["Momentum"] = np.tanh(df["Momentum"].fillna(0))

# ---------------------------
# ML dataset builder & training (if enabled)
# ---------------------------
def build_ml_dataset(tickers, lookback_days=250, future_days=20):
    X_list = []
    y_list = []
    for t in tickers:
        try:
            s_all = get_close_series(t)
            if s_all.empty: continue
        except Exception:
            continue
        if len(s_all) < (future_days + 60):
            continue
        for i in range(60, len(s_all) - future_days):
            window = s_all.iloc[i-60:i]
            # features
            ret5 = (s_all.iloc[i-1] - s_all.iloc[i-1-5]) / (s_all.iloc[i-1-5] + 1e-9) if i-1-5 >= 0 else 0.0
            ret20 = (s_all.iloc[i-1] - s_all.iloc[i-1-20]) / (s_all.iloc[i-1-20] + 1e-9) if i-1-20 >= 0 else 0.0
            ret60 = (s_all.iloc[i-1] - s_all.iloc[i-1-60]) / (s_all.iloc[i-1-60] + 1e-9) if i-1-60 >= 0 else 0.0
            vol20 = window.pct_change().dropna().tail(20).std() if len(window) >= 20 else np.nan
            ma5 = window.tail(5).mean()
            ma20 = window.tail(20).mean()
            delta = window.diff().dropna()
            up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1] if not delta.empty else 0.0
            down = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1] if not delta.empty else 0.0
            rs = up / (down + 1e-9)
            rsi14 = 100 - (100 / (1 + rs))
            future_ret = (s_all.iloc[i+future_days] - s_all.iloc[i-1]) / (s_all.iloc[i-1] + 1e-9)
            target = 1 if (future_ret * 100.0) >= target_pct else 0
            X_list.append([ret5, ret20, ret60, vol20, ma5/(ma20+1e-9), (ma5-ma20)/(ma20+1e-9), rsi14])
            y_list.append(target)
    X = pd.DataFrame(X_list, columns=["ret5","ret20","ret60","vol20","ma5_ma20","ma_diff_norm","rsi14"])
    return X, np.array(y_list)

# train ML if enabled
ml_prob_map = {}  # ticker -> prob
if enable_ml:
    st.subheader("ML: building dataset & training (this can be slow)...")
    ml_universe = universe if (max_tickers_for_ml in (0, None) or max_tickers_for_ml <= 0) else universe[:max_tickers_for_ml]
    with st.spinner("Building ML dataset..."):
        X, y = build_ml_dataset(ml_universe, lookback_days=250, future_days=20)
    if X.shape[0] < 200:
        st.warning("Not enough ML training rows collected. ML will be skipped. Increase universe or history.")
        enable_ml = False
    else:
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_imp = imp.fit_transform(X)
        Xs = scaler.fit_transform(X_imp)
        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y if y.sum()>0 else None)
        clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
        with st.spinner("Training ML model..."):
            clf.fit(Xtr, ytr)
        test_score = clf.score(Xte, yte) if len(yte)>0 else None
        st.success(f"ML trained. Test accuracy ≈ {round(test_score*100,2) if test_score is not None else 'N/A'}%")

        # Predict probabilities for each ticker
        st.subheader("ML: predicting current probability for universe")
        for t in ml_universe:
            s = get_close_series(t)
            if s.empty or len(s) < 60:
                ml_prob_map[t] = 0.0
                continue
            # latest features
            ret5 = (s.iloc[-1] - s.iloc[-1-5]) / (s.iloc[-1-5] + 1e-9) if len(s) > 5 else 0.0
            ret20 = (s.iloc[-1] - s.iloc[-1-20]) / (s.iloc[-1-20] + 1e-9) if len(s) > 20 else 0.0
            ret60 = (s.iloc[-1] - s.iloc[-1-60]) / (s.iloc[-1-60] + 1e-9) if len(s) > 60 else 0.0
            window = s.iloc[-60:]
            vol20 = window.pct_change().dropna().tail(20).std() if len(window) >= 20 else np.nan
            ma5 = window.tail(5).mean()
            ma20 = window.tail(20).mean()
            delta = window.diff().dropna()
            up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1] if not delta.empty else 0.0
            down = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1] if not delta.empty else 0.0
            rs = up / (down + 1e-9)
            rsi14 = 100 - (100 / (1 + rs))
            feat = np.array([[ret5, ret20, ret60, vol20, ma5/(ma20+1e-9), (ma5-ma20)/(ma20+1e-9), rsi14]])
            feat_imp = imp.transform(feat)
            feat_s = scaler.transform(feat_imp)
            prob = clf.predict_proba(feat_s)[0,1]
            ml_prob_map[t] = float(prob)
else:
    # ML disabled: fill zeros
    for t in universe:
        ml_prob_map[t] = 0.0

# ---------------------------
# Combine into Final Score & Expected Price
# ---------------------------
# Normalize weights
total_w = w_momentum + w_sentiment + w_geo + w_ml
if total_w == 0:
    total_w = 1.0
w_m = w_momentum / total_w
w_s = w_sentiment / total_w
w_g = w_geo / total_w
w_machine = w_ml / total_w

out_rows = []
for _, row in df.iterrows():
    t = row["Ticker"]
    price = row["Price"]
    momentum = row["Momentum"]
    sent = row["SentScore"]
    geo = row["GeoScore"]
    mlp = ml_prob_map.get(t, 0.0)
    final_score = (w_m * momentum) + (w_s * sent) + (w_g * geo) + (w_machine * mlp)
    # Expected price: current * (1 + Prob * target_pct)
    expected_price = round(price * (1 + mlp * (target_pct / 100.0)), 2) if (not np.isnan(price) and price is not None) else np.nan
    out_rows.append({
        "Ticker": t,
        "Company": row["Company"],
        "Price": price,
        "Momentum": round(momentum,4),
        "Sentiment": round(sent,4),
        "GeoScore": round(geo,4),
        "ML_ProbUp1M": round(mlp,4),
        "FinalScore": round(final_score,4),
        "ExpectedPrice_1M": expected_price
    })

out_df = pd.DataFrame(out_rows).sort_values(by="FinalScore", ascending=False).reset_index(drop=True)

st.subheader("Hybrid Ensemble Results (sorted by FinalScore)")
visible = ["Ticker","Company","Price","FinalScore","Momentum","Sentiment","GeoScore","ML_ProbUp1M","ExpectedPrice_1M"]
st.dataframe(out_df[visible].fillna("N/A"), use_container_width=True)

# ---------------------------
# Shortlist: price <= 500 and FinalScore >= threshold
# ---------------------------
shortlist = out_df[(out_df["Price"] <= 500) & (out_df["FinalScore"] >= finalscore_threshold)].sort_values(by="FinalScore", ascending=False)
st.subheader(f"🔎 Shortlist (Price ≤ ₹500 AND FinalScore ≥ {finalscore_threshold})")
if shortlist.empty:
    st.info("No stocks meet the shortlist criteria. Try lowering thresholds or increasing universe.")
else:
    st.dataframe(shortlist[visible].reset_index(drop=True), use_container_width=True)
    st.download_button("Download Shortlist CSV", data=shortlist.to_csv(index=False).encode(), file_name="hybrid_shortlist.csv", mime="text/csv")

# ---------------------------
# Charts
# ---------------------------
st.subheader("Charts")
topn = min(20, len(out_df))
chart_df = out_df.head(topn).dropna(subset=["Price"])
if not chart_df.empty:
    fig, ax = plt.subplots(figsize=(12,4))
    ax.bar(chart_df["Ticker"], chart_df["FinalScore"])
    ax.set_ylabel("Final Score")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# ---------------------------
# Full download
# ---------------------------
st.subheader("Download Full Dataset")
buf = BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    out_df.to_excel(writer, index=False, sheet_name="HybridScores")
st.download_button("Download Excel (Full)", data=buf.getvalue(), file_name=f"hybrid_scores_{date.today()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Notes: ML and sentiment may be slow for large universes. Use 'Max tickers' or turn off sentiment for faster runs.")
