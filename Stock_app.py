import requests

def get_nse_price(symbol: str):
    """
    Fetch live stock price from NSE India website
    :param symbol: NSE stock symbol (e.g. 'TATAPOWER', 'RELIANCE')
    :return: dict with last price, change, % change
    """
    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}"
    }

    session = requests.Session()
    response = session.get(url, headers=headers, timeout=10)

    if response.status_code == 200:
        data = response.json()
        price_info = data.get("priceInfo", {})
        return {
            "symbol": symbol,
            "lastPrice": price_info.get("lastPrice"),
            "change": price_info.get("change"),
            "pChange": price_info.get("pChange")
        }
    else:
        raise Exception(f"Failed to fetch data (status {response.status_code})")

# -----------------------------
# Example: Fetch live price
# -----------------------------
if __name__ == "__main__":
    symbol = "TATAPOWER"   # Change this to any NSE symbol
    try:
        result = get_nse_price(symbol)
        print(f"✅ {result['symbol']}")
        print(f"Last Price: {result['lastPrice']}")
        print(f"Change: {result['change']} ({result['pChange']}%)")
    except Exception as e:
        print("⚠️ Error:", e)
