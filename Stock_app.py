import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Step 1: Tiny ticker list (5 Midcaps for hybrid test)
# -----------------------------
tickers = ["ABB.NS", "BANKBARODA.NS", "CANBK.NS", "TATAPOWER.NS", "TVSMOTOR.NS"]
print(f"✅ Loaded {len(tickers)} tickers for hybrid test")

# -----------------------------
# Step 2: Parameters
# -----------------------------
start = datetime.datetime.now() - datetime.timedelta(days=90)   # 3 months history
end = datetime.datetime.now()
target_pct = 5
all_results = []

# -----------------------------
# Step 3: Process each ticker
# -----------------------------
for i, ticker in enumerate(tickers, start=1):
    try:
        print(f"({i}/{len(tickers)}) Downloading {ticker} ...")
        df = yf.download(ticker, start=start, end=end, progress=True)
        if df.empty:
            continue

        # Features
        df["Return_5d"] = df["Adj Close"].pct_change(5)
        df["Return_20d"] = df["Adj Close"].pct_change(20)
        df["Volatility"] = df["Adj Close"].pct_change().rolling(10).std()
        delta = df["Adj Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))

        # Target
        df["Target"] = (df["Adj Close"].shift(-10) / df["Adj Close"] - 1) * 100
        df["Target"] = (df["Target"] >= target_pct).astype(int)
        df.dropna(inplace=True)

        if len(df) < 30:
            continue

        features = ["Return_5d","Return_20d","Volatility","RSI"]
        X = df[features]
        y = df["Target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Models
        xgb = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                            use_label_encoder=False, eval_metric="logloss", random_state=42)
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        lr = LogisticRegression(max_iter=500)

        xgb.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        lr.fit(X_train, y_train)

        # Probabilities
        p_xgb = xgb.predict_proba(X.iloc[-1:].values)[0][1]
        p_rf = rf.predict_proba(X.iloc[-1:].values)[0][1]
        p_lr = lr.predict_proba(X.iloc[-1:].values)[0][1]

        # Hybrid probability (weighted average)
        hybrid_prob = (0.5 * p_xgb) + (0.3 * p_rf) + (0.2 * p_lr)

        # Current price
        current_price = round(df["Adj Close"].iloc[-1], 2)
        expected_price_1m = round(current_price * (1 + hybrid_prob * (target_pct/100)), 2)

        all_results.append({
            "Company": ticker,
            "Price": current_price,
            "HybridProb": round(hybrid_prob, 3),
            "ExpectedPrice_1M": expected_price_1m
        })

    except Exception as e:
        print(f"⚠️ Error processing {ticker}: {e}")

# -----------------------------
# Step 4: Results
# -----------------------------
pred_df = pd.DataFrame(all_results)
if not pred_df.empty:
    print("\n✅ Hybrid Predictions Complete")
    print(pred_df)
else:
    print("⚠️ No data processed.")
