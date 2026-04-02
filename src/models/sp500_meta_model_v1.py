import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor
from src.strategy.position_sizing1 import compute_position_sizes


# =========================
# LOAD
# =========================

df = pd.read_csv("data/sp500_multi_horizon.csv")
df = df.dropna().reset_index(drop=True)


# =========================
# FEATURES BASE
# =========================

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


# =========================
# SPLIT
# =========================

split = int(len(X) * 0.7)

X_train, X_test = X[:split], X[split:]

returns = df["return"].values
returns_test = returns[split:]


# =========================
# 1️⃣ MULTI-HORIZON SIGNAL
# =========================

targets = {
    "5d": df["target_5d"],
    "21d": df["target_21d"],
    "63d": df["target_63d"]
}

preds_dict = {}

for name, y in targets.items():

    y_train = y[:split]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds_dict[name] = model.predict(X_test)


multi_signal = (
    0.2 * preds_dict["5d"] +
    0.5 * preds_dict["21d"] +
    0.3 * preds_dict["63d"]
)


# =========================
# 2️⃣ QUANTILE MODEL
# =========================

y_train = df["target_21d"][:split]

quantiles = [0.1, 0.5, 0.9]
preds_q = {}

for q in quantiles:

    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=q,
        n_estimators=200,
        max_depth=6
    )

    model.fit(X_train, y_train)
    preds_q[q] = model.predict(X_test)


q10 = preds_q[0.1]
q50 = preds_q[0.5]
q90 = preds_q[0.9]

uncertainty = q90 - q10


# =========================
# 3️⃣ META FEATURES
# =========================

meta_df = pd.DataFrame({
    "multi_signal": multi_signal,
    "expected": q50,
    "uncertainty": uncertainty,
    "vol": df["vol_20"].values[split:],
    "trend": df["trend_200"].values[split:]
})


# target meta = futuro ritorno
meta_target = df["target_21d"].values[split:]


# =========================
# 4️⃣ META MODEL
# =========================

meta_model = lgb.LGBMRegressor(
    n_estimators=300,
    max_depth=4
)

meta_model.fit(meta_df, meta_target)

meta_score = meta_model.predict(meta_df)


# =========================
# 🔴 FINAL SIGNAL
# =========================

final_signal = multi_signal * (1 + meta_score * 5)


# =========================
# POSITION SIZING
# =========================

sizes = compute_position_sizes(final_signal, df)


# =========================
# BACKTEST
# =========================

equity = 1.0
equity_curve = []

for i in range(len(sizes) - 1):
    equity *= (1 + sizes[i] * returns_test[i + 1])
    equity_curve.append(equity)

equity_curve = np.array(equity_curve)


# =========================
# BUY & HOLD
# =========================

bh_equity = 1.0
bh_curve = []

for r in returns_test:
    bh_equity *= (1 + r)
    bh_curve.append(bh_equity)

bh_curve = np.array(bh_curve)


# =========================
# PERFORMANCE
# =========================

max_drawdown = np.min(equity_curve / np.maximum.accumulate(equity_curve) - 1)

print("\n=== META MODEL ===")
print("Strategy equity:", equity)
print("Buy&Hold equity:", bh_equity)
print("Max drawdown:", max_drawdown)


# =========================
# PLOT
# =========================

plt.figure(figsize=(10, 5))
plt.plot(equity_curve, label="Meta Strategy")
plt.plot(bh_curve, label="Buy & Hold")
plt.legend()
plt.title("Meta Model V3")
plt.show()