import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import datetime, requests
from io import BytesIO
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from openpyxl.formatting.rule import ColorScaleRule

# -----------------------------
# Step 1: Fetch Midcap + Smallcap tickers dynamically
# -----------------------------
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.niftyindices.com/"}

def fetch_index_tickers(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(BytesIO(r.content))
    sym_col = [c for c in df.columns if "Symbol" in c or "SYMBOL" in c][0]
    tickers = df[sym_col].astype(str).str.strip().str.upper() + ".NS"
    return tickers.tolist()

midcap_url = "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv"
smallcap_url = "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap100list.csv"

try:
    midcap_tickers = fetch_index_tickers(midcap_url)
    smallcap_tickers = fetch_index_tickers(smallcap_url)
except Exception as e:
    print("⚠️ Could not fetch live tickers, using fallback list:", e)
    midcap_tickers = ["ABB.NS","AUROPHARMA.NS","BANKBARODA.NS","CANBK.NS"]
    smallcap_tickers = ["ABFRL.NS","CESC.NS","FORTIS.NS","IRCTC.NS"]

tickers = midcap_tickers + smallcap_tickers
print(f"✅ Loaded {len(midcap_tickers)} Midcap and {len(smallcap_tickers)} Smallcap tickers")

# -----------------------------
# Step 2: Parameters
# -----------------------------
start = datetime.datetime.now() - datetime.timedelta(days=365*2)
end = datetime.datetime.now()
target_pct = 5   # Target return % in 1 month
prob_threshold = 0.6  # ML probability filter
price_limit = 1000    # Shortlist price limit

all_results = []

# -----------------------------
# Step 3: Process each ticker
# -----------------------------
for ticker in tickers:
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            continue

        # Features
        df["Return_5d"] = df["Adj Close"].pct_change(5)
        df["Return_20d"] = df["Adj Close"].pct_change(20)
        df["Volatility"] = df["Adj Close"].pct_change().rolling(20).std()

        # RSI (simplified)
        delta = df["Adj Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))

        # Target variable
        df["Target"] = (df["Adj Close"].shift(-20) / df["Adj Close"] - 1) * 100
        df["Target"] = (df["Target"] >= target_pct).astype(int)
        df.dropna(inplace=True)

        if len(df) < 50:
            continue

        # ML
        features = ["Return_5d", "Return_20d", "Volatility", "RSI"]
        X = df[features]
        y = df["Target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        latest_features = X.iloc[-1:].values
        prob_up = model.predict_proba(latest_features)[0][1]

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

        # Final Score (blend ML + Momentum20 + RSI_norm)
        rsi_norm = (df["RSI"].iloc[-1] / 100) if not np.isnan(df["RSI"].iloc[-1]) else 0.5
        momentum20 = df["Return_20d"].iloc[-1] if not np.isnan(df["Return_20d"].iloc[-1]) else 0
        final_score = (0.5 * prob_up) + (0.25 * momentum20) + (0.25 * rsi_norm)

        all_results.append({
            "Company": ticker,
            "Price": current_price,
            "ExpectedPrice_1M": expected_price_1m,
            "ExpectedPrice_1Y": expected_price_1y,
            "Diff_1M": diff_1m,
            "Diff_1Y": diff_1y,
            "PctChange_1M": pctchange_1m,
            "PctChange_1Y": pctchange_1y,
            "Momentum_5d": round(df["Return_5d"].iloc[-1], 3),
            "Momentum_20d": round(momentum20, 3),
            "Volatility": round(df["Volatility"].iloc[-1], 3),
            "RSI": round(df["RSI"].iloc[-1], 2),
            "ProbUp1M": round(prob_up, 3),
            "FinalScore": round(final_score, 3)
        })
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# -----------------------------
# Step 4: Output
# -----------------------------
pred_df = pd.DataFrame(all_results)

if not pred_df.empty:
    # Rank by FinalScore
    pred_df["Rank"] = pred_df["FinalScore"].rank(ascending=False, method="dense").astype(int)

    # Column order
    cols = ["Company","Price","ExpectedPrice_1M","ExpectedPrice_1Y",
            "Diff_1M","Diff_1Y","PctChange_1M","PctChange_1Y","FinalScore","Rank"] + \
           [c for c in pred_df.columns if c not in ["Company","Price","ExpectedPrice_1M","ExpectedPrice_1Y",
                                                    "Diff_1M","Diff_1Y","PctChange_1M","PctChange_1Y",
                                                    "FinalScore","Rank"]]
    pred_df = pred_df[cols]

    # Shortlist
    shortlist = pred_df[(pred_df["Price"] <= price_limit) & (pred_df["ProbUp1M"] >= prob_threshold)]
    shortlist = shortlist.sort_values(by="FinalScore", ascending=False)

    # Save CSV
    pred_df.to_csv("all_predictions.csv", index=False)
    shortlist.to_csv("shortlist.csv", index=False)

    # Save Excel with formatting
    excel_file = "predictions.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        pred_df.to_excel(writer, sheet_name="All Predictions", index=False)
        shortlist.to_excel(writer, sheet_name="Shortlist", index=False)

    wb = load_workbook(excel_file)
    for sheet_name in ["All Predictions","Shortlist"]:
        ws = wb[sheet_name]

        # Red/Green fill for PctChange columns
        for col_letter in ["G","H"]:  # PctChange_1M, PctChange_1Y
            ws.conditional_formatting.add(
                f"{col_letter}2:{col_letter}{ws.max_row}",
                ColorScaleRule(start_type="num", start_value=-50, start_color="FF0000",
                               mid_type="num", mid_value=0, mid_color="FFFFFF",
                               end_type="num", end_value=50, end_color="00FF00")
            )

        # Gradient for FinalScore
        ws.conditional_formatting.add(
            f"I2:I{ws.max_row}",
            ColorScaleRule(start_type="num", start_value=0, start_color="FF0000",
                           mid_type="num", mid_value=0.5, mid_color="FFFF00",
                           end_type="num", end_value=1, end_color="00FF00")
        )

    wb.save(excel_file)

    print("✅ Analysis complete!")
    print("\n--- All Predictions (sample) ---")
    print(pred_df.head(10))
    print("\n--- Shortlist (sample) ---")
    print(shortlist.head(10))
    print(f"\n📊 Excel with formatting saved as {excel_file}")
else:
    print("No data processed.")
