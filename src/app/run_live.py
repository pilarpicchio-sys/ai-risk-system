import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor
from src.strategy.position_sizing1 import compute_position_sizes
from src.optimization.cvar_portfolio import compute_cvar_weights


# =========================
# ASSETS
# =========================

assets = {
    "sp500": "^GSPC",
    "gold": "GC=F",
    "bonds": "TLT"
}


# =========================
# BUILD DATA LIVE
# =========================

def build_live_dataset(ticker):

    df = yf.download(ticker, start="2000-01-01")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].rename(columns={"Close": "price"})
    df = df.reset_index()

    df["return"] = df["price"].pct_change()

    df["momentum_5"] = df["price"].pct_change(5)
    df["momentum_20"] = df["price"].pct_change(20)
    df["momentum_60"] = df["price"].pct_change(60)

    df["vol_20"] = df["return"].rolling(20).std()
    df["vol_60"] = df["return"].rolling(60).std()

    df["trend_50"] = df["price"] / df["price"].rolling(50).mean()
    df["trend_200"] = df["price"] / df["price"].rolling(200).mean()

    df["zscore_20"] = (
        df["price"] - df["price"].rolling(20).mean()
    ) / df["price"].rolling(20).std()

    # targets (servono per training interno)
    for h in [5, 21, 63]:
        df[f"target_{h}d"] = df["price"].pct_change(h).shift(-h)

    df = df.dropna().reset_index(drop=True)

    return df


# =========================
# MODEL + SIGNAL
# =========================

def compute_signal(df):

    feature_cols = [
        "momentum_5",
        "momentum_20",
        "momentum_60",
        "vol_20",
        "vol_60",
        "trend_50",
        "trend_200",
        "zscore_20"
    ]

    X = df[feature_cols]

    split = int(len(df) * 0.8)

    X_train, X_test = X[:split], X[split:]
    df_test = df.iloc[split:].reset_index(drop=True)

    # MULTI-HORIZON
    preds_dict = {}

    for h in ["5d", "21d", "63d"]:

        y_train = df[f"target_{h}"][:split]

        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=6,
            random_state=42
        )

        model.fit(X_train, y_train)
        preds_dict[h] = model.predict(X_test)

    multi_signal = (
        0.2 * preds_dict["5d"] +
        0.5 * preds_dict["21d"] +
        0.3 * preds_dict["63d"]
    )

    # QUANTILE
    y_train = df["target_21d"][:split]

    preds_q = {}

    for q in [0.1, 0.5, 0.9]:

        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=150,
            max_depth=6
        )

        model.fit(X_train, y_train)
        preds_q[q] = model.predict(X_test)

    q10 = preds_q[0.1]
    q50 = preds_q[0.5]
    q90 = preds_q[0.9]

    uncertainty = q90 - q10

    # META
    meta_df = pd.DataFrame({
        "multi_signal": multi_signal,
        "expected": q50,
        "uncertainty": uncertainty,
        "vol": df_test["vol_20"].values,
        "trend": df_test["trend_200"].values
    })

    meta_target = df["target_21d"][split:]

    meta_model = lgb.LGBMRegressor(
        n_estimators=150,
        max_depth=4
    )

    meta_model.fit(meta_df, meta_target)
    meta_score = meta_model.predict(meta_df)

    meta_score = np.clip(meta_score, -0.3, 0.3)

    final_signal = multi_signal * (1 + meta_score * 1.2)

    # prendi solo ultimo segnale
    return final_signal[-1], df_test


# =========================
# RUN LIVE
# =========================

signals = []
dfs = []

for name, ticker in assets.items():

    print(f"Processing {name}...")

    df = build_live_dataset(ticker)
    signal, df_test = compute_signal(df)

    signals.append(signal)
    dfs.append(df_test)


# =========================
# POSITION SIZING
# =========================

sizes = []

for signal, df_test in zip(signals, dfs):

    s = compute_position_sizes(np.array([signal]), df_test)[0]
    sizes.append(s)

sizes = np.array(sizes)


# =========================
# CVAR WEIGHTS
# =========================

returns_list = []

for df in dfs:
    r = df["return"].values[-200:]
    returns_list.append(r)

returns_matrix = np.column_stack(returns_list)

weights = compute_cvar_weights(returns_matrix)

# =========================
# OUTPUT
# =========================

print("\n=== LIVE SIGNALS ===")

for i, name in enumerate(assets.keys()):
    print(f"{name}: signal={signals[i]:.4f} | size={sizes[i]:.3f} | weight={weights[i]:.2f}")

print("\nPortfolio allocation:")
print(weights)


# =========================
# CAPITAL ALLOCATION (€)
# =========================

capital = 10000  # cambia qui il tuo capitale

print("\n=== CAPITAL ALLOCATION (€) ===")

for i, name in enumerate(assets.keys()):

    alloc = capital * weights[i] * sizes[i]

    print(f"{name}: €{alloc:.2f}")
    

# =========================
# ALERT SYSTEM
# =========================

print("\n=== ALERTS ===")

triggered = False

for i, name in enumerate(assets.keys()):

    if sizes[i] > 0.25:

        triggered = True

        direction = "LONG" if signals[i] > 0 else "REDUCE"

        print(f"⚠️ {name.upper()} | {direction} | size={sizes[i]:.2f}")

if not triggered:
    print("No strong signals today.")

    import json
import os

os.makedirs("reports/data", exist_ok=True)

data = {
    "assets": list(assets.keys()),
    "signals": list(signals),
    "sizes": list(sizes),
    "weights": list(weights)
}

with open("reports/data/live_signals.json", "w") as f:
    json.dump(data, f)
