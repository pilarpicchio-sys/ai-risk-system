import numpy as np
import matplotlib.pyplot as plt

from src.data.load_data import load_csv
from src.features.returns import compute_returns
from src.features.multi_horizon import compute_multi_horizon_returns
from src.features.basic_features import add_basic_features
from src.data.prepare_dataset import prepare_dataset

from src.models.simple_model import train_model


# =========================
# LOAD DATA
# =========================

df = load_csv("data/gold.csv")

df = compute_returns(df)
df = compute_multi_horizon_returns(df)
df = add_basic_features(df)


# =========================
# FEATURES
# =========================

df["momentum_10"] = df["price"].pct_change(10)
df["momentum_20"] = df["price"].pct_change(20)

df["vol_20"] = df["return"].rolling(20).std()
df["vol_50"] = df["return"].rolling(50).std()

df["trend_50"] = df["price"].pct_change(50)

df["zscore_20"] = (
    df["price"] - df["price"].rolling(20).mean()
) / df["price"].rolling(20).std()

# 🎯 TARGET GIUSTO
df["target"] = (df["return"].shift(-1) > 0).astype(int)

df = df.dropna().reset_index(drop=True)

print("Dataset ready:", df.shape)


# =========================
# FEATURES LIST
# =========================

feature_cols = [
    "return",
    "log_return",
    "rolling_mean_2",
    "rolling_vol_2",
    "momentum_2",
    "momentum_10",
    "momentum_20",
    "vol_20",
    "trend_50",
    "zscore_20"
]

feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols]
y = df["target"]

print("Columns in X:", X.columns.tolist())


# =========================
# TRAIN / TEST
# =========================

split = int(len(X) * 0.7)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

returns = df["return"].values
returns_test = returns[split:]


# =========================
# MODEL
# =========================

model = train_model(X_train, y_train)

probs = model.predict_proba(X_test)
long_prob = probs[:, 1]

print("\nProbabilities sample:")
print(long_prob[:5])


# =========================
# DECISION FUNCTION
# =========================

import numpy as np
import matplotlib.pyplot as plt

from src.data.load_data import load_csv
from src.features.returns import compute_returns
from src.features.multi_horizon import compute_multi_horizon_returns
from src.features.basic_features import add_basic_features
from src.data.prepare_dataset import prepare_dataset

from src.models.simple_model import train_model


# =========================
# LOAD DATA
# =========================

df = load_csv("data/gold.csv")

df = compute_returns(df)
df = compute_multi_horizon_returns(df)
df = add_basic_features(df)


# =========================
# FEATURES
# =========================

df["momentum_10"] = df["price"].pct_change(10)
df["momentum_20"] = df["price"].pct_change(20)

df["vol_20"] = df["return"].rolling(20).std()
df["vol_50"] = df["return"].rolling(50).std()

df["trend_50"] = df["price"].pct_change(50)

df["zscore_20"] = (
    df["price"] - df["price"].rolling(20).mean()
) / df["price"].rolling(20).std()

# 🎯 TARGET GIUSTO
df["target"] = (df["return"].shift(-1) > 0).astype(int)

df = df.dropna().reset_index(drop=True)

print("Dataset ready:", df.shape)


# =========================
# FEATURES LIST
# =========================

feature_cols = [
    "return",
    "log_return",
    "rolling_mean_2",
    "rolling_vol_2",
    "momentum_2",
    "momentum_10",
    "momentum_20",
    "vol_20",
    "trend_50",
    "zscore_20"
]

feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols]
y = df["target"]

print("Columns in X:", X.columns.tolist())


# =========================
# TRAIN / TEST
# =========================

split = int(len(X) * 0.7)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

returns = df["return"].values
returns_test = returns[split:]


# =========================
# MODEL
# =========================

model = train_model(X_train, y_train)

probs = model.predict_proba(X_test)
long_prob = probs[:, 1]

print("\nProbabilities sample:")
print(long_prob[:5])


# =========================
# DECISION FUNCTION
# =========================

def make_decisions(probs, df):

    sizes = []

    momentum = df["momentum_10"].values[-len(probs):]
    vol = df["vol_20"].values[-len(probs):]
    trend = df["trend_50"].values[-len(probs):]

    for i in range(len(probs)):

        p = probs[i]
        edge = p - 0.5

        # 🔥 EDGE FILTER
        if edge < 0.02:
            sizes.append(0.0)
            continue

        # base sizing
        size = edge * 5

        # volatility targeting
        if vol[i] > 0:
            size *= (0.02 / vol[i])

        # trend boost
        if trend[i] > 0:
            size *= 1.3
        else:
            size *= 0.7

        # momentum boost
        if momentum[i] > 0:
            size *= 1.2
        else:
            size *= 0.8

        # clamp
        size = max(0.0, min(size, 1.5))

        sizes.append(size)

    return np.array(sizes)


sizes = make_decisions(long_prob, df)

print("\nSizes sample:")
print(sizes[:10])


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
plt.plot(equity_curve, label="Strategy (TEST)")
plt.plot(bh_curve, label="Buy & Hold (TEST)")
plt.title("Out-of-sample Equity Curve")
plt.legend()
plt.show()


sizes = make_decisions(long_prob, df)

print("\nSizes sample:")
print(sizes[:10])


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
plt.plot(equity_curve, label="Strategy (TEST)")
plt.plot(bh_curve, label="Buy & Hold (TEST)")
plt.title("Out-of-sample Equity Curve")
plt.legend()
plt.show()