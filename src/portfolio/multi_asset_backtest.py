import numpy as np
import pandas as pd

from src.strategy.position_sizing1 import compute_position_sizes
from src.app.generate_pro_chart import generate_pro_chart


# =========================
# LOAD
# =========================

def load_asset(path):
    df = pd.read_csv(path)
    df = df.dropna().reset_index(drop=True)
    return df


# =========================
# SINGLE ASSET PIPELINE
# =========================

def run_single_asset(df):

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
    returns = df["return"].values

    split = int(len(df) * 0.7)

    X_train, X_test = X[:split], X[split:]
    df_test = df.iloc[split:].reset_index(drop=True)
    returns_test = returns[split:]

    # =========================
    # SIGNAL (semplice)
    # =========================

    signal = df["target_21d"].values[split:]

    sizes = compute_position_sizes(signal, df_test)

    # =========================
    # EQUITY
    # =========================

    equity = 1.0
    curve = []

    for i in range(len(sizes) - 1):
        size = min(sizes[i], 1.0)
        equity *= (1 + size * returns_test[i + 1])
        curve.append(equity)

    return np.array(curve)


# =========================
# RUN MULTI-ASSET
# =========================

assets = ["sp500", "gold", "bonds"]

equities = []

for name in assets:
    print(f"Running {name}...")
    df = load_asset(f"data/{name}_multi_horizon.csv")
    eq = run_single_asset(df)
    equities.append(eq)


# =========================
# ALIGN
# =========================

min_len = min(len(e) for e in equities)
equities = [e[-min_len:] for e in equities]

# =========================
# WEIGHTS (EQUAL)
# =========================

weights = np.array([1 / len(equities)] * len(equities))

portfolio_curve = np.zeros(min_len)

for w, e in zip(weights, equities):
    portfolio_curve += w * e


# =========================
# BENCHMARK (SP500)
# =========================

df_sp = load_asset("data/sp500_multi_horizon.csv")

returns_sp = df_sp["return"].values
split = int(len(df_sp) * 0.7)
returns_sp = returns_sp[split:]

bench = 1.0
bench_curve = []

for r in returns_sp:
    bench *= (1 + r)
    bench_curve.append(bench)

bench_curve = np.array(bench_curve)

# ALIGN BENCH
min_len = min(len(portfolio_curve), len(bench_curve))
portfolio_curve = portfolio_curve[-min_len:]
bench_curve = bench_curve[-min_len:]


# =========================
# METRICS
# =========================

def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1
    return np.min(dd)


print("\n=== PORTFOLIO ===")
print("Final equity:", portfolio_curve[-1])
print("Max drawdown:", max_drawdown(portfolio_curve))

print("\n=== BENCHMARK ===")
print("Final equity:", bench_curve[-1])
print("Max drawdown:", max_drawdown(bench_curve))

# =========================
# SAVE RESULTS
# =========================

import os

os.makedirs("reports/data", exist_ok=True)

np.save("reports/data/portfolio_curve.npy", portfolio_curve)
np.save("reports/data/bench_curve.npy", bench_curve)


# =========================
# CHART
# =========================

from src.app.generate_pro_chart import generate_pro_chart_fast


chart_path = generate_pro_chart_fast()

print("\nChart saved:", chart_path)

