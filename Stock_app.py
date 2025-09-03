import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from io import BytesIO
from GoogleNews import GoogleNews
from textblob import TextBlob
import matplotlib.pyplot as plt

st.set_page_config(page_title="Indian Midcap+Smallcap — Dynamic", layout="wide")
st.title("📊 Indian Midcap + Smallcap Predictor (Dynamic NSE lists)")

# ------------------------------
# Helpers to fetch NSE constituents (dynamic)
# ------------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.niftyindices.com/"
}

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
    """
    which: 'MIDCAP100' or 'SMALLCAP100'
    Returns DataFrame with columns: ['Symbol','Company Name','Ticker']
    """
    urls = INDEX_URLS[which]
    last_err = None
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            # Try common CSV shapes
            df = pd.read_csv(BytesIO(r.content))
            cols = {c.lower(): c for c in df.columns}
            # Normalize expected columns
            sym_col = cols.get("symbol") or cols.get("ticker") or cols.get("symbol \n") or list(df.columns)[0]
            name_col = cols.get("company name") or cols.get("companyname") or cols.get("name") or list(df.columns)[1]
            out = df[[sym_col, name_col]].copy()
            out.columns = ["Symbol", "Company Name"]
            out["Symbol"] = out["Symbol"].astype(str).str.strip().str.upper()
            out["Ticker"] = out["Symbol"].apply(lambda s: f"{s}.NS")
            # Drop any duplicates/empty
            out = out.dropna(subset=["Symbol"]).drop_duplicates(subset=["Symbol"])
            return out.reset_index(drop=True)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to fetch {which} list. Last error: {last_err}")

# ------------------------------
# News sentiment
# ------------------------------
@st.cache_data(ttl=6*60*60)
def company_sentiment(company_name: str, max_heads: int = 5) -> float:
    """
    Returns average TextBlob polarity for top headlines in past day.
    Range roughly [-1, +1]. 0 = neutral.
    """
    try:
        g = GoogleNews(lang="en", period="1d")
        g.search(company_name + " India")
        news = g.result()[:max_heads]
        if not news:
            return 0.0
        scores = []
        for n in news:
            title = n.get("title") or ""
            if title:
                scores.append(TextBlob(title).sentiment.polarity)
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))
    except Exception:
        return 0.0

def label_from_polarity(p: float) -> str:
    if p > 0.1:
        return "Positive"
    if p < -0.1:
        return "Negative"
    return "Neutral"

# ------------------------------
# Sidebar controls
# ------------------------------
st.sidebar.header("Settings")
index_choice = st.sidebar.multiselect(
    "Select indices",
    ["MIDCAP100", "SMALLCAP100"],
    default=["MIDCAP100", "SMALLCAP100"]
)

calc_news = st.sidebar.checkbox("Compute news sentiment (slower)", value=True)
max_headlines = st.sidebar.slider("Headlines per company", 1, 8, 4)
top_n_chart = st.sidebar.slider("Bar chart: show top N by % change", 5, 30, 15)

st.sidebar.caption("Tip: Turn off sentiment to load faster.")

# ------------------------------
# Fetch lists + build universe
# ------------------------------
universe = []
names = {}
try:
    if "MIDCAP100" in index_choice:
        df_mid = fetch_index_constituents("MIDCAP100")
        universe += df_mid["Ticker"].tolist()
        names.update(dict(zip(df_mid["Ticker"], df_mid["Company Name"])))
    if "SMALLCAP100" in index_choice:
        df_small = fetch_index_constituents("SMALLCAP100")
        universe += df_small["Ticker"].tolist()
        names.update(dict(zip(df_small["Ticker"], df_small["Company Name"])))
except Exception as e:
    st.error(f"Could not fetch index constituents. {e}")
    st.stop()

universe = sorted(set(universe))
st.write(f"*Universe size:* {len(universe)} tickers")

if not universe:
    st.warning("No tickers selected.")
    st.stop()

# ------------------------------
# Fetch price data (vectorized)
# ------------------------------
st.subheader("📈 Daily % Change")
with st.spinner("Downloading prices…"):
    data = yf.download(universe, period="2d", interval="1d", group_by="ticker", threads=True, auto_adjust=False)

def pct_change_for(t: str):
    try:
        s = data[t]["Close"].dropna()
        if len(s) < 2:
            return None
        return round(((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2]) * 100, 2)
    except Exception:
        return None

rows = []
for t in universe:
    change = pct_change_for(t)
    comp_name = names.get(t, t.replace(".NS",""))
    if calc_news:
        pol = company_sentiment(comp_name, max_heads=max_headlines)
        sent = label_from_polarity(pol)
    else:
        pol, sent = 0.0, "Neutral"
    rows.append({
        "Company": comp_name,
        "Symbol": t.replace(".NS",""),
        "Ticker": t,
        "Change % (Last Day)": change,
        "Sentiment Score": round(pol, 3),
        "Sentiment": sent
    })

df = pd.DataFrame(rows)
df = df.sort_values(by=["Change % (Last Day)"], ascending=False, na_position="last").reset_index(drop=True)

st.dataframe(df, use_container_width=True)

# ------------------------------
# Charts
# ------------------------------
clean = df.dropna(subset=["Change % (Last Day)"]).head(top_n_chart)
if not clean.empty:
    st.subheader(f"📊 Top {len(clean)} by % Change")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(clean["Symbol"], clean["Change % (Last Day)"])
    ax.axhline(0, linestyle="--")
    ax.set_ylabel("%")
    ax.set_xlabel("Symbol")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# ------------------------------
# Download as Excel
# ------------------------------
st.subheader("📥 Download")
out = BytesIO()
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Mid+Small (Dynamic)")
st.download_button(
    "Download Excel",
    data=out.getvalue(),
    file_name="mid_small_dynamic_predictions.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("Constituents are fetched live from NSE/Nifty Indices CSV endpoints and cached for 24 hours.")
