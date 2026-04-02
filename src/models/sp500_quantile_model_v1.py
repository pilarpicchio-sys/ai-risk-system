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
# QUANTILES
# =========================

quantiles = [0.1, 0.5, 0.9]
models = {}
preds_q = {}


for q in quantiles:

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=int(q * 100)
    )

    # hack semplice: pesi per simulare quantile
    weights = np.where(y_train > y_train.quantile(q), 2, 1)

    model.fit(X_train, y_train, sample_weight=weights)

    preds = model.predict(X_test)

    models[q] = model
    preds_q[q] = preds

    print(f"Quantile {q} trained")


# =========================
# DISTRIBUTION METRICS
# =========================

q10 = preds_q[0.1]
q50 = preds_q[0.5]
q90 = preds_q[0.9]

expected = q50
downside = q10
upside = q90


# =========================
# DECISION (TAIL-AWARE)
# =========================

def make_decisions(expected, downside, df):

    sizes = []

    vol = df["vol_20"].values[-len(expected):]

    for i in range(len(expected)):

        edge = expected[i] - downside[i]

        if edge < 0:
            sizes.append(0.0)
            continue

        size = edge * 80

        if vol[i] > 0:
            size *= (0.02 / vol[i])

        size = max(0.0, min(size, 1.5))

        sizes.append(size)

    return np.array(sizes)


sizes = make_decisions(expected, downside, df)


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

print("\n=== QUANTILE MODEL PERFORMANCE ===")
print("Strategy equity:", equity)
print("Buy&Hold equity:", bh_equity)
print("Max drawdown:", max_drawdown)


# =========================
# PLOT
# =========================

plt.figure(figsize=(10, 5))
plt.plot(equity_curve, label="Quantile Strategy")
plt.plot(bh_curve, label="Buy & Hold")
plt.legend()
plt.title("Distribution-based Strategy")
plt.show()