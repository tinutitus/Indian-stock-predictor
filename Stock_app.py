import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# Step 1: Hardcoded Top 20 Midcap tickers
# -----------------------------
tickers = [
    "ABB.NS","AUROPHARMA.NS","BANKBARODA.NS","CANBK.NS","CHOLAFIN.NS",
    "CUMMINSIND.NS","GODREJPROP.NS","INDHOTEL.NS","JSWENERGY.NS","LICHSGFIN.NS",
    "MUTHOOTFIN.NS","OBEROIRLTY.NS","PEL.NS","PNB.NS","RECLTD.NS",
    "SRF.NS","TATAPOWER.NS","TVSMOTOR.NS","UNIONBANK.NS","YESBANK.NS"
]

print(f"✅ Loaded {len(tickers)} Midcap tickers")

# -----------------------------
# Step 2: Parameters
# -----------------------------
start = datetime.datetime.now() - datetime.timedelta(days=365)  # 1 year history
end = datetime.datetime.now()
target_pct = 5
prob_threshold = 0.6
price_limit = 1000

all_results = []

# -----------------------------
# Step 3: Process each ticker
# -----------------------------
for i, ticker in enumerate(tickers, start=1):
    try:
        print(f"({i}/{len(tickers)}) Processing {ticker}...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            continue

        # Technical indicators
        df["Return_5d"] = df["Adj Close"].pct_change(5)
        df["Return_20d"] = df["Adj Close"].pct_change(20)
        df["Volatility"] = df["Adj Close"].pct_change().rolling(20).std()

        # Moving Averages
        df["MA20"] = df["Adj Close"].rolling(20).mean()
        df["MA50"] = df["Adj Close"].rolling(50).mean()

        # RSI
        delta = df["Adj Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))

        # Target
        df["Target"] = (df["Adj Close"].shift(-20) / df["Adj Close"] - 1) * 100
        df["Target"] = (df["Target"] >= target_pct).astype(int)
        df.dropna(inplace=True)

        if len(df) < 60:
            continue

        features = ["Return_5d","Return_20d","Volatility","MA20","MA50","RSI"]
        X = df[features]
        y = df["Target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              use_label_encoder=False, eval_metric="logloss",
                              random_state=42)
        model.fit(X_train, y_train)

        prob_up = model.predict_proba(X.iloc[-1:].values)[0][1]
        current_price = round(df["Adj Close"].iloc[-1], 2)

        # Expected Prices
        expected_price_1m = round(current_price * (1 + prob_up * (target_pct/100)), 2)
        expected_price_1y = round(current_price * ((1 + prob_up * (target_pct/100)) ** 12), 2)

        # Differences
        diff_1m = round(expected_price_1m - current_price, 2)
        diff_1y = round(expected_price_1y - current_price, 2)

        # Percentage Changes
        pctchange_1m = round((diff_1m / current_price) * 100, 2) if current_price else 0
        pctchange_1y = round((diff_1y / current_price) * 100, 2) if current_price else 0

        # Final Score
        rsi_norm = (df["RSI"].iloc[-1] / 100) if not np.isnan(df["RSI"].iloc[-1]) else 0.5
        momentum20 = df["Return_20d"].iloc[-1] if not np.isnan(df["Return_20d"].iloc[-1]) else 0
        final_score = (0.6 * prob_up) + (0.2 * momentum20) + (0.2 * rsi_norm)

        all_results.append({
            "Company": ticker,
            "Price": current_price,
            "ExpectedPrice_1M": expected_price_1m,
            "ExpectedPrice_1Y": expected_price_1y,
            "Diff_1M": diff_1m,
            "Diff_1Y": diff_1y,
            "PctChange_1M": pctchange_1m,
            "PctChange_1Y": pctchange_1y,
            "RSI": round(df["RSI"].iloc[-1], 2),
            "Momentum_20d": round(momentum20, 3),
            "ProbUp1M": round(prob_up, 3),
            "FinalScore": round(final_score, 3)
        })
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# -----------------------------
# Step 4: Save Excel
# -----------------------------
pred_df = pd.DataFrame(all_results)

if not pred_df.empty:
    pred_df["Rank"] = pred_df["FinalScore"].rank(ascending=False, method="dense").astype(int)

    pred_df = pred_df.sort_values(by="Rank")

    excel_file = "predictions.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        pred_df.to_excel(writer, sheet_name="All Predictions", index=False)

    print("✅ Analysis complete!")
    print(f"📊 Excel saved as {excel_file}")
else:
    print("⚠️ No data processed.")
