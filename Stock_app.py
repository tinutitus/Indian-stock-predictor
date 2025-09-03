import requests
import pandas as pd
import datetime

# -----------------------------
# Step 1: Fetch NSE data for 1 stock
# -----------------------------
symbol = "TATAPOWER"
url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
}

session = requests.Session()
response = session.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    last_price = data["priceInfo"]["lastPrice"]
    change = data["priceInfo"]["change"]
    pchange = data["priceInfo"]["pChange"]

    print(f"✅ {symbol}")
    print(f"Last Price: {last_price}")
    print(f"Change: {change} ({pchange}%)")
else:
    print(f"⚠️ Failed to fetch data for {symbol} | Status: {response.status_code}")
