# ---------------------------
# ONE-MONTH ML FORECAST: Predict next ~20 trading day (1-month) return > target_pct
# ---------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.subheader("🔮 One-month ML Forecast (screen for investable stocks ≤ ₹500)")

# UI inputs
target_pct = st.sidebar.number_input("Target return in 1 month (%)", value=5.0, step=0.5)
prob_threshold = st.sidebar.slider("Probability threshold for shortlist", 0.50, 0.95, 0.65, 0.05)
max_tickers_for_ml = st.sidebar.number_input("Max tickers to use for ML (0=all)", min_value=0, value=0, step=10)

# Respect user limit
ml_universe = universe if (max_tickers_for_ml in (0, None) or max_tickers_for_ml <= 0) else universe[:max_tickers_for_ml]

# Helper: build feature rows and targets across tickers
def build_ml_dataset(tickers, lookback_days=250, future_days=20):
    X_list = []
    y_list = []
    tick_map = []  # to map rows back to ticker & date
    for t in tickers:
        # get close series
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
        # create rolling features for each eligible day i where we have past 60 days and future future_days
        for i in range(60, len(s_all) - future_days):
            window = s_all.iloc[i-60:i]  # last 60 days up to i-1
            # features
            ret_5 = (s_all.iloc[i-1] - s_all.iloc[i-1-5]) / s_all.iloc[i-1-5] if i-1-5 >= 0 else 0.0
            ret_20 = (s_all.iloc[i-1] - s_all.iloc[i-1-20]) / s_all.iloc[i-1-20] if i-1-20 >= 0 else 0.0
            ret_60 = (s_all.iloc[i-1] - s_all.iloc[i-1-60]) / s_all.iloc[i-1-60] if i-1-60 >= 0 else 0.0
            vol20 = window.pct_change().dropna().tail(20).std() if len(window) >= 20 else np.nan
            ma5 = window.tail(5).mean()
            ma20 = window.tail(20).mean()
            ma50 = window.tail(50).mean() if len(window) >= 50 else np.nan
            # RSI 14 using simple method on window
            delta = window.diff().dropna()
            up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1] if not delta.empty else 0.0
            down = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1] if not delta.empty else 0.0
            rs = up / (down + 1e-9)
            rsi14 = 100 - (100 / (1 + rs))
            # future return
            future_ret = (s_all.iloc[i+future_days] - s_all.iloc[i-1]) / s_all.iloc[i-1]
            target = 1 if (future_ret * 100.0) >= target_pct else 0
            X_list.append([ret_5, ret_20, ret_60, vol20, ma5/ (ma20+1e-9), (ma5-ma20)/(ma20+1e-9), rsi14])
            y_list.append(target)
            tick_map.append((t, s_all.index[i-1]))  # associate row with ticker and date used
    X = pd.DataFrame(X_list, columns=["ret_5","ret_20","ret_60","vol20","ma5_ma20_ratio","ma_diff_norm","rsi14"])
    y = np.array(y_list)
    return X, y, tick_map

# Build dataset (this may take time)
with st.spinner("Building ML training dataset (this can take time for many tickers)..."):
    X, y, tick_map = build_ml_dataset(ml_universe, lookback_days=250, future_days=20)

if X.shape[0] < 200:
    st.warning("Not enough training rows collected. Try increasing max tickers or lower target / use more history.")
else:
    # train/test split
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imp.fit_transform(X)
    Xs = scaler.fit_transform(X_imp)

    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y if y.sum()>0 else None)

    # Use RandomForestClassifier for robust classification
    clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    with st.spinner("Training ML model..."):
        clf.fit(Xtr, ytr)
    test_score = clf.score(Xte, yte) if len(yte)>0 else None
    st.success(f"ML model trained. Test accuracy ≈ {round(test_score*100,2) if test_score is not None else 'N/A'}%")

    # Now predict current-up-probability for each ticker using most recent features
    preds = []
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
            preds.append(np.nan)
            continue
        # build latest features using last available index i = len(s)-1
        ret_5 = (s.iloc[-1] - s.iloc[-1-5]) / s.iloc[-1-5] if len(s)>5 else 0.0
        ret_20 = (s.iloc[-1] - s.iloc[-1-20]) / s.iloc[-1-20] if len(s)>20 else 0.0
        ret_60 = (s.iloc[-1] - s.iloc[-1-60]) / s.iloc[-1-60] if len(s)>60 else 0.0
        window = s.iloc[-60:]
        vol20 = window.pct_change().dropna().tail(20).std() if len(window)>=20 else np.nan
        ma5 = window.tail(5).mean()
        ma20 = window.tail(20).mean()
        ma50 = window.tail(50).mean() if len(window)>=50 else np.nan
        delta = window.diff().dropna()
        up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1] if not delta.empty else 0.0
        down = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1] if not delta.empty else 0.0
        rs = up / (down + 1e-9)
        rsi14 = 100 - (100 / (1 + rs))
        feat = np.array([[ret_5, ret_20, ret_60, vol20, ma5/(ma20+1e-9), (ma5-ma20)/(ma20+1e-9), rsi14]])
        feat_imp = imp.transform(feat)
        feat_s = scaler.transform(feat_imp)
        prob = clf.predict_proba(feat_s)[0,1]
        preds.append(prob)
        pred_rows.append({"Ticker": t, "Company": name_map.get(t,t.replace(".NS","")), "ProbUp1M": round(float(prob),4)})
    pred_df = pd.DataFrame(pred_rows)
    # merge latest close price to filter price <= 500
    latest_price = {}
    for t in pred_df["Ticker"].tolist():
        try:
            s = price_data[t]["Close"].dropna()
            latest_price[t] = float(s.iloc[-1]) if len(s)>0 else np.nan
        except Exception:
            latest_price[t] = np.nan
    pred_df["Price"] = pred_df["Ticker"].map(latest_price)
    # shortlist: price <= 500 and prob >= threshold
    shortlist = pred_df[(pred_df["Price"] <= 500) & (pred_df["ProbUp1M"] >= prob_threshold)].sort_values(by="ProbUp1M", ascending=False)
    st.markdown("### 🔎 ML shortlist (price ≤ ₹500 & prob ≥ threshold)")
    if shortlist.empty:
        st.info("No stocks meet the criteria. Try lowering probability threshold, increasing max tickers for ML, or increasing target return.")
    else:
        st.dataframe(shortlist.reset_index(drop=True), use_container_width=True)
        # Offer download
        st.download_button("Download shortlist (CSV)", data=shortlist.to_csv(index=False).encode(), file_name="ml_shortlist.csv", mime="text/csv")
