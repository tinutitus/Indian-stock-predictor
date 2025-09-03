import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import datetime

# -----------------------------
# Step 1: Choose stock
# -----------------------------
symbol = "TATAPOWER.NS"   # Change this to any NSE stock
print(f"📊 Running analysis for {symbol}")

# -----------------------------
# Step 2: Download last 1 year of data
# -----------------------------
start = datetime.datetime.now() - datetime.timedelta(days=365)
end = datetime.datetime.now()

df = yf.download(symbol, start=start, end=end, progress=False)

if df.empty:
    print("⚠️ No data found!")
    exit()

# Flatten MultiIndex if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]

# Ensure Adj Close exists
if "Adj Close" not in df.columns:
    if "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    else:
        print("⚠️ No Adj Close or Close column available")
        exit()

# -----------------------------
# Step 3: Features
# -----------------------------
df["Return_5d"] = df["Adj Close"].pct_change(5)
df["Return_20d"] = df["Adj Close"].pct_change(20)
df["Volatility"] = df["Adj Close"].pct_change().rolling(20).std()

# RSI
delta = df["Adj Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / (loss + 1e-9)
df["RSI"] = 100 - (100 / (1 + rs))

# Target: will price go up by 5% in 20 days?
target_pct = 5
df["Target"] = (df["Adj Close"].shift(-20) / df["Adj Close"] - 1) * 100
df["Target"] = (df["Target"] >= target_pct).astype(int)
df.dropna(inplace=True)

if len(df) < 50:
    print("⚠️ Not enough data for ML training")
    exit()

# -----------------------------
# Step 4: Train ML Model
# -----------------------------
features = ["Return_5d","Return_20d","Volatility","RSI"]
X = df[features]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction for last day
latest_features = X.iloc[-1:].values
prob_up = model.predict_proba(latest_features)[0][1]

# -----------------------------
# Step 5: Price predictions
# -----------------------------
current_price = round(df["Adj Close"].iloc[-1], 2)
expected_price_1m = round(current_price * (1 + prob_up * (target_pct/100)), 2)
expected_price_1y = round(current_price * ((1 + prob_up * (target_pct/100)) ** 12), 2)

print("\n✅ Results")
print(f"Current Price: {current_price}")
print(f"Expected Price (1M): {expected_price_1m}")
print(f"Expected Price (1Y): {expected_price_1y}")
print(f"Probability of going UP next month: {round(prob_up,3)}")

# -----------------------------
# Step 6: Save results
# -----------------------------
results = pd.DataFrame([{
    "Company": symbol,
    "Current Price": current_price,
    "Expected Price (1M)": expected_price_1m,
    "Expected Price (1Y)": expected_price_1y,
    "Prob Up (1M)": round(prob_up,3)
}])

results.to_csv("stock_prediction.csv", index=False)
results.to_excel("stock_prediction.xlsx", index=False)

print("\n📂 Saved results to stock_prediction.csv and stock_prediction.xlsx")
