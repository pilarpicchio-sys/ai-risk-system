import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor


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
# MODEL
# =========================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

preds_z = (preds - preds.mean()) / preds.std()


# =========================
# 🔴 REGIME FEATURE
# =========================

vol = df["vol_20"]
vol_mean = vol.rolling(252).mean()

df["vol_regime"] = vol / vol_mean


# =========================
# DECISION
# =========================

def make_decisions(preds_z, df):

    sizes = []

    trend = df["trend_200"].values[-len(preds_z):]
    vol = df["vol_20"].values[-len(preds_z):]
    regime = df["vol_regime"].values[-len(preds_z):]

    for i in range(len(preds_z)):

        z = preds_z[i]

        base = 0.6
        alpha = z * 0.35

        size = base + alpha

        # conviction
        if z > 1:
            size *= 1.3
        elif z < -1:
            size *= 0.5

        # 🔴 VOL REGIME CONTROL
        if regime[i] > 1.2:
            size *= 0.6
        elif regime[i] < 0.8:
            size *= 1.1

        # trend
        if trend[i] < 1:
            size *= 0.6

        # vol scaling
        if vol[i] > 0:
            size *= (0.018 / vol[i])

        size = max(0.0, min(size, 1.5))

        sizes.append(size)

    return np.array(sizes)


sizes = make_decisions(preds_z, df)


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

print("\n=== TEST PERFORMANCE ===")
print("Strategy equity:", equity)
print("Buy&Hold equity:", bh_equity)
print("Max drawdown:", max_drawdown)


# =========================
# PLOT
# =========================

plt.figure(figsize=(10, 5))
plt.plot(equity_curve, label="Strategy V8")
plt.plot(bh_curve, label="Buy & Hold")
plt.legend()
plt.title("Regime-Aware Strategy")
plt.show()