import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# LOAD
# =========================

df = pd.read_csv("data/macro_dataset.csv")
df = df.dropna().reset_index(drop=True)


# =========================
# 🔴 CORE SIGNAL (SPREAD)
# =========================

# normalizziamo scale diverse
df["gold_norm"] = df["gold_price"] / df["gold_price"].rolling(252).mean()
df["rates_norm"] = df["rates_price"] / df["rates_price"].rolling(252).mean()

# spread macro vero
df["macro_spread"] = df["gold_norm"] - df["rates_norm"]


# =========================
# Z-SCORE DELLO SPREAD
# =========================

mean = df["macro_spread"].rolling(252).mean()
std = df["macro_spread"].rolling(252).std()

df["spread_z"] = (df["macro_spread"] - mean) / std


df = df.dropna().reset_index(drop=True)


# =========================
# DECISION RULE (NO ML)
# =========================

sizes = []

for z in df["spread_z"]:

    # gold sottovalutato vs tassi
    if z < -1:
        sizes.append(1.0)

    # neutro
    elif -1 <= z <= 1:
        sizes.append(0.3)

    # gold caro vs tassi
    else:
        sizes.append(0.0)

sizes = np.array(sizes)


# =========================
# BACKTEST
# =========================

returns = df["gold_ret"].values

equity = 1.0
equity_curve = []

for i in range(len(sizes) - 1):
    equity *= (1 + sizes[i] * returns[i + 1])
    equity_curve.append(equity)

equity_curve = np.array(equity_curve)


# =========================
# BUY & HOLD
# =========================

bh_equity = 1.0
bh_curve = []

for r in returns:
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
plt.plot(equity_curve, label="Macro Spread Strategy")
plt.plot(bh_curve, label="Buy & Hold")
plt.legend()
plt.title("Gold vs Real Yield Proxy")
plt.show()