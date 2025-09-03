# Stock_app.py
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

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Stock Intelligence (Nifty50 + Midcap + Smallcap)", layout="wide")
st.title("🔬 Stock Intelligence — Technicals + Sentiment + Geo + 1M ML Forecast")

# ---------------------------
# Index CSV endpoints (try multiple mirrors)
# ---------------------------
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.niftyindices.com/"}
INDEX_URLS = {
    "NIFTY50": [
        "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
        "https://www1.nseindia.com/content/indices/ind_nifty50list.csv",
    ],
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
# Helper: fetch constituents (cached)
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
st.sidebar.header("Universe & Options")
indices = st.sidebar.multiselect("Include indices", ["NIFTY50", "MIDCAP100", "SMALLCAP100"], default=["NIFTY50","MIDCAP100","SMALLCAP100"])
limit_universe = st.sidebar.number_input("Max tickers (0 = all)", min_value=0, value=0, step=10)
calc_sentiment = st.sidebar.checkbox("Compute news sentiment (slower)", value=True)
headlines_per_company = st.sidebar.slider("Headlines per company", min_value=1, max_value=6, value=3)
top_n_chart = st.sidebar.slider("Top N for bar chart", 5, 30, 15)

st.sidebar.markdown("### Scoring weights (normalized internally)")
w_momentum = st.sidebar.slider("Momentum weight", 0.0, 1.0, 0.4, 0.05)
w_news = st.sidebar.slider("Company news weight", 0.0, 1.0, 0.3, 0.05)
w_geo = st.sidebar.slider("Geopolitics weight", 0.0, 1.0, 0.3, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### ML Forecast (1-month)")
enable_ml_ui = st.sidebar.checkbox("Enable ML shortlist UI", value=True)
target_pct = st.sidebar.number_input("Target return in 1 month (%)", value=5.0, step=0.5)
prob_threshold = st.sidebar.slider("Probability threshold for shortlist", 0.50, 0.95, 0.65, 0.05)
max_tickers_for_ml = st.sidebar.number_input("Max tickers to use for ML (0=all)", min_value=0, value=0, step=10)

st.sidebar.markdown("---")
st.sidebar.write("Tip: turn off sentiment to speed up when testing. Limit tickers for quick tests.")

# ---------------------------
# Geopolitical topics (global) used for GeoScore
# ---------------------------
geo_topics = ["Oil prices", "US Fed rates", "India-China relations", "Global inflation"]

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

# ---------------------------
# Build the universe dynamically
# ---------------------------
st.info("Fetching index constituents (this may take a few seconds)...")
universe = []
name_map = {}
try:
    if "NIFTY50" in indices:
        dfn = fetch_index_constituents("NIFTY50")
        universe += dfn["Ticker"].tolist()
        name_map.update(dict(zip(dfn["Ticker"], dfn["Company Name"])))
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

st.write(f"Universe size: **{len(universe)}**")
if not universe:
    st.warning("No tickers selected. Choose indices in the sidebar.")
    st.stop()

# ---------------------------
# Download price history (vectorized)
# ---------------------------
days_needed = 300  # to compute MA200 and ML features
st.subheader("Downloading price history...")
with st.spinner("Downloading prices from Yahoo Finance (this can take ~30-90s for many tickers)..."):
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
# Technical indicators functions
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
# Precompute global topic sentiments
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

    # Company news sentiment
    comp_sent = fetch_company_sentiment(comp_name, max_heads=headlines_per_company) if calc_sentiment else 0.0

    # Geopolitics heuristics
    geo_vals = []
    if any(k in comp_name.upper() for k in ["OIL","PETRO","GAS","ENERGY","LNG","PETRONET"]):
        geo_vals.append(topic_scores.get("Oil prices", 0.0))
    if any(k in tk for k in ["BANK", "HDFC", "ICICI", "AXIS", "KOTAK", "PNB", "YES"]):
        geo_vals.append(topic_scores.get("US Fed rates", 0.0))
    if any(k in comp_name.upper() for k in ["INFRA", "EXPORT", "AUTOMOTIVE", "AUTO", "TATA", "MAHINDRA", "MARUTI"]):
        geo_vals.append(topic_scores.get("India-China relations", 0.0))
    if not geo_vals:
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
# Momentum: z-score normalize tech inputs and compose
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
# FinalScore (weights normalized)
# ---------------------------
total_w = w_momentum + w_news + w_geo
if total_w == 0:
    total_w = 1.0
w_news_n = w_news / total_w
w_mom_n = w_momentum / total_w
w_geo_n = w_geo / total_w

df["FinalScore"] = (w_mom_n * df["Momentum"]) + (w_news_n * df["SentScore"]) + (w_geo_n * df["GeoScore"])
df = df.sort_values(by="FinalScore", ascending=False).reset_index(drop=True)

# ---------------------------
# Show results table & charts
# ---------------------------
st.subheader("Results — sorted by Final Score")
visible_cols = ["Ticker","Company","FinalScore","Momentum","SentScore","GeoScore","LastDay%"]
st.dataframe(df[visible_cols].fillna("N/A"), use_container_width=True)

# Top N by LastDay% bar chart
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

# Final score distribution
fig2, ax2 = plt.subplots(figsize=(6,3))
ax2.hist(df["FinalScore"].dropna(), bins=30)
ax2.set_xlabel("Final Score")
st.pyplot(fig2)

# ---------------------------
# ONE-MONTH ML FORECAST
# ---------------------------
if enable_ml_ui:
    st.subheader("🔮 One-month ML Forecast (screen for investable stocks ≤ ₹500)")
    st.write("This builds a pooled training set across tickers and trains a RandomForest classifier.")
    # UI values already captured above: target_pct, prob_threshold, max_tickers_for_ml

    # Respect user limit
    ml_universe = universe if (max_tickers_for_ml in (0, None) or max_tickers_for_ml <= 0) else universe[:max_tickers_for_ml]

    def build_ml_dataset(tickers, lookback_days=250, future_days=20):
        X_list = []
        y_list = []
        tick_map = []
        for t in tickers:
            try:
                s_all = price_data[t]["Close"].dropna()
            except Exception:
                try:
                    tmp = yf.download(t, period=f"{lookback_days+future_days+10}d", interval="1d", progress=False)
                    s_all = tmp["Close"].dropna()
                except Exception:
                    continue
            if len(s_all) < (future_days + 60):
                continue
            for i in range(60, len(s_all) - future_days):
                # window of past 60 days ending at i-1
                window = s_all.iloc[i-60:i]
                ret_5 = (s_all.iloc[i-1] - s_all.iloc[i-1-5]) / s_all.iloc[i-1-5] if i-1-5 >= 0 else 0.0
                ret_20 = (s_all.iloc[i-1] - s_all.iloc[i-1-20]) / s_all.iloc[i-1-20] if i-1-20 >= 0 else 0.0
                ret_60 = (s_all.iloc[i-1] - s_all.iloc[i-1-60]) / s_all.iloc[i-1-60] if i-1-60 >= 0 else 0.0
                vol20 = window.pct_change().dropna().tail(20).std() if len(window) >= 20 else np.nan
                ma5 = window.tail(5).mean()
                ma20 = window.tail(20).mean()
                ma50 = window.tail(50).mean() if len(window) >= 50 else np.nan
                delta = window.diff().dropna()
                up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1] if not delta.empty else 0.0
                down = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1] if not delta.empty else 0.0
                rs = up / (down + 1e-9)
                rsi14 = 100 - (100 / (1 + rs))
                future_ret = (s_all.iloc[i+future_days] - s_all.iloc[i-1]) / s_all.iloc[i-1]
                target = 1 if (future_ret * 100.0) >= target_pct else 0
                X_list.append([ret_5, ret_20, ret_60, vol20, ma5/(ma20+1e-9), (ma5-ma20)/(ma20+1e-9), rsi14])
                y_list.append(target)
                tick_map.append((t, s_all.index[i-1]))
        X = pd.DataFrame(X_list, columns=["ret_5","ret_20","ret_60","vol20","ma5_ma20_ratio","ma_diff_norm","rsi14"])
        y = np.array(y_list)
        return X, y, tick_map

    with st.spinner("Building ML training dataset (this can take time for many tickers)..."):
        X, y, tick_map = build_ml_dataset(ml_universe, lookback_days=250, future_days=20)

    if X.shape[0] < 200:
        st.warning("Not enough training rows collected. Try increasing max tickers or lower target / use more history.")
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
        st.success(f"ML model trained. Test accuracy ≈ {round(test_score*100,2) if test_score is not None else 'N/A'}%")

        # Predict current probability for each ticker
        pred_rows = []
        for t in ml_universe:
            try:
                s = price_data[t]["Close"].dropna()
            except Exception:
                try:
                    df_tmp = yf.download(t, period="500d", interval="1d", progress=False)
                    s = df_tmp["Close"].dropna()
                except Exception:
                    s = pd.Series(dtype=float)
            if len(s) < 60:
                continue
            ret_5 = (s.iloc[-1] - s.iloc[-1-5]) / s.iloc[-1-5] if len(s) > 5 else 0.0
            ret_20 = (s.iloc[-1] - s.iloc[-1-20]) / s.iloc[-1-20] if len(s) > 20 else 0.0
            ret_60 = (s.iloc[-1] - s.iloc[-1-60]) / s.iloc[-1-60] if len(s) > 60 else 0.0
            window = s.iloc[-60:]
            vol20 = window.pct_change().dropna().tail(20).std() if len(window) >= 20 else np.nan
            ma5 = window.tail(5).mean()
            ma20 = window.tail(20).mean()
            delta = window.diff().dropna()
            up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1] if not delta.empty else 0.0
            down = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1] if not delta.empty else 0.0
            rs = up / (down + 1e-9)
            rsi14 = 100 - (100 / (1 + rs))
            feat = np.array([[ret_5, ret_20, ret_60, vol20, ma5/(ma20+1e-9), (ma5-ma20)/(ma20+1e-9), rsi14]])
            feat_imp = imp.transform(feat)
            feat_s = scaler.transform(feat_imp)
            prob = clf.predict_proba(feat_s)[0,1]
            pred_rows.append({"Ticker": t, "Company": name_map.get(t,t.replace(".NS","")), "ProbUp1M": round(float(prob),4)})

        pred_df = pd.DataFrame(pred_rows)
        latest_price = {}
        for t in pred_df["Ticker"].tolist():
            try:
                s = price_data[t]["Close"].dropna()
                latest_price[t] = float(s.iloc[-1]) if len(s) > 0 else np.nan
            except Exception:
                latest_price[t] = np.nan
        pred_df["Price"] = pred_df["Ticker"].map(latest_price)

        shortlist = pred_df[(pred_df["Price"] <= 500) & (pred_df["ProbUp1M"] >= prob_threshold)].sort_values(by="ProbUp1M", ascending=False)
        st.markdown("### 🔎 ML shortlist (price ≤ ₹500 & prob ≥ threshold)")
        if shortlist.empty:
            st.info("No stocks meet the criteria. Try lowering probability threshold, increasing max tickers for ML, or increasing target return.")
        else:
            st.dataframe(shortlist.reset_index(drop=True), use_container_width=True)
            st.download_button("Download shortlist (CSV)", data=shortlist.to_csv(index=False).encode(), file_name="ml_shortlist.csv", mime="text/csv")

# ---------------------------
# Download full results
# ---------------------------
st.subheader("Download full dataset")
out = BytesIO()
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Scores")
st.download_button("Download Excel (Full)", data=out.getvalue(), file_name=f"stock_scores_{date.today()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Notes: Company & topic sentiment may slow down runs. Use 'Max tickers' to test quickly. ML training can take several minutes depending on universe size.")
