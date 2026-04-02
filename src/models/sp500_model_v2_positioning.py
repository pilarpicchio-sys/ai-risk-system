import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/sp500_dataset.csv")
df = df.dropna().reset_index(drop=True)

print("Data loaded:", df.shape)


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

print("Train size:", len(X_train))
print("Test size:", len(X_test))


# =========================
# MODEL
# =========================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Preds length:", len(preds))


# =========================
# DECISION
# =========================

def make_decisions(preds, df):

    sizes = []

    trend = df["trend_200"].values[-len(preds):]
    vol = df["vol_20"].values[-len(preds):]

    for i in range(len(preds)):

        p = preds[i]

        if p < 0.002:
            sizes.append(0.0)
            continue

        size = p * 50

        if trend[i] < 1:
            size *= 0.3

        if vol[i] > 0:
            size *= (0.02 / vol[i])

        size = max(0.0, min(size, 1.5))

        sizes.append(size)

    return np.array(sizes)


sizes = make_decisions(preds, df)

print("Sizes length:", len(sizes))


# =========================
# BACKTEST
# =========================

equity = 1.0
equity_curve = []

for i in range(len(sizes) - 1):
    equity *= (1 + sizes[i] * returns_test[i + 1])
    equity_curve.append(equity)

equity_curve = np.array(equity_curve)

print("Equity curve length:", len(equity_curve))


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
plt.plot(equity_curve, label="Strategy")
plt.plot(bh_curve, label="Buy & Hold")
plt.legend()
plt.title("SP500 Strategy")
plt.show()