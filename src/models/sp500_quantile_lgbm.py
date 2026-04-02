import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from src.strategy.position_sizing1 import compute_position_sizes


# =========================
# LOAD
# =========================

df = pd.read_csv("data/sp500_dataset.csv")
df = df.dropna().reset_index(drop=True)


# =========================
# FEATURES
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
y = df["target"]


# =========================
# SPLIT
# =========================

split = int(len(X) * 0.7)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

returns = df["return"].values
returns_test = returns[split:]


# =========================
# QUANTILE MODELS
# =========================

quantiles = [0.1, 0.5, 0.9]
preds_q = {}

for q in quantiles:

    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=q,
        n_estimators=300,
        max_depth=6
    )

    model.fit(X_train, y_train)

    preds_q[q] = model.predict(X_test)

    print(f"Quantile {q} trained")


# =========================
# DISTRIBUTION
# =========================

q10 = preds_q[0.1]
q50 = preds_q[0.5]
q90 = preds_q[0.9]

expected = q50
uncertainty = q90 - q10


# =========================
# 🔴 CONFIDENCE
# =========================

confidence = 1 / (1 + uncertainty * 50)

final_signal = expected * confidence


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

print("\n=== LGBM QUANTILE ===")
print("Strategy equity:", equity)
print("Buy&Hold equity:", bh_equity)
print("Max drawdown:", max_drawdown)


# =========================
# PLOT
# =========================

plt.figure(figsize=(10, 5))
plt.plot(equity_curve, label="LGBM Quantile")
plt.plot(bh_curve, label="Buy & Hold")
plt.legend()
plt.title("True Distribution Model")
plt.show()