import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from src.strategy.position_sizing1 import compute_position_sizes


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/sp500_multi_horizon.csv")
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


# =========================
# TARGETS
# =========================

targets = {
    "5d": df["target_5d"],
    "21d": df["target_21d"],
    "63d": df["target_63d"]
}


# =========================
# SPLIT
# =========================

split = int(len(X) * 0.7)

X_train, X_test = X[:split], X[split:]

returns = df["return"].values
returns_test = returns[split:]


# =========================
# TRAIN MODELS
# =========================

models = {}
preds_dict = {}

for name, y in targets.items():

    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    models[name] = model
    preds_dict[name] = preds

    print(f"{name} model trained")


# =========================
# 🔴 COMBINE SIGNALS
# =========================

# pesi (semplici ma efficaci)
weights = {
    "5d": 0.2,
    "21d": 0.5,
    "63d": 0.3
}

combined_preds = np.zeros(len(preds_dict["21d"]))

for name in preds_dict:
    combined_preds += weights[name] * preds_dict[name]

    


# =========================
# POSITION SIZING (V3 CORE)
# =========================

sizes = compute_position_sizes(combined_preds, df)


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

print("\n=== MULTI-HORIZON PERFORMANCE ===")
print("Strategy equity:", equity)
print("Buy&Hold equity:", bh_equity)
print("Max drawdown:", max_drawdown)


# =========================
# PLOT
# =========================

plt.figure(figsize=(10, 5))
plt.plot(equity_curve, label="Multi-Horizon Strategy")
plt.plot(bh_curve, label="Buy & Hold")
plt.legend()
plt.title("Multi-Horizon V3")
plt.show()