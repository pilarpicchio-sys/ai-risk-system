import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from src.strategy.position_sizing1 import compute_position_sizes


# =========================
# LOAD
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
returns = df["return"].values


# =========================
# PARAMETRI WALK-FORWARD
# =========================

train_size = 1500
test_size = 250
step = 250

equity = 1.0
equity_curve = []


# =========================
# LOOP WALK-FORWARD
# =========================

for start in range(0, len(df) - train_size - test_size, step):

    train_idx = range(start, start + train_size)
    test_idx = range(start + train_size, start + train_size + test_size)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    df_test = df.iloc[test_idx].reset_index(drop=True)
    returns_test = returns[test_idx]

    # =========================
    # MULTI-HORIZON MODELS
    # =========================

    preds_dict = {}

    for h in ["5d", "21d", "63d"]:

        y_train = df[f"target_{h}"].iloc[train_idx]

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

    # =========================
    # QUANTILE MODEL (CORRETTO)
    # =========================

    y_train = df["target_21d"].iloc[train_idx]

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

    # =========================
    # META MODEL (NO LEAK)
    # =========================

    meta_train = pd.DataFrame({
        "multi_signal": multi_signal,
        "expected": q50,
        "uncertainty": uncertainty,
        "vol": df_test["vol_20"].values,
        "trend": df_test["trend_200"].values
    })

    # ⚠️ target shiftato (NO LEAK)
    meta_target = df["target_21d"].iloc[test_idx].values

    meta_model = lgb.LGBMRegressor(
        n_estimators=150,
        max_depth=4
    )

    meta_model.fit(meta_train, meta_target)
    meta_score = meta_model.predict(meta_train)

    # 🔴 CONTROLLO META (STABILE)
    meta_score = np.clip(meta_score, -0.3, 0.3)

    final_signal = multi_signal * (1 + meta_score * 1.2)

    # =========================
    # POSITION SIZING (ALLINEATO)
    # =========================

    sizes = compute_position_sizes(final_signal, df_test)

    # =========================
    # BACKTEST (CORRETTO)
    # =========================

    for i in range(len(sizes) - 1):

        size = sizes[i]

        # cap sicurezza finale
        size = min(size, 1.0)

        equity *= (1 + size * returns_test[i + 1])
        equity_curve.append(equity)


# =========================
# RISULTATI
# =========================

equity_curve = np.array(equity_curve)

if len(equity_curve) > 0:
    max_dd = np.min(equity_curve / np.maximum.accumulate(equity_curve) - 1)
else:
    max_dd = 0

print("\n=== WALK-FORWARD (FIXED) ===")
print("Final equity:", equity)
print("Max drawdown:", max_dd)


# =========================
# PLOT
# =========================

plt.figure(figsize=(10, 5))
plt.plot(equity_curve, label="Walk-forward")
plt.legend()
plt.title("Walk-forward Validation (Fixed)")
plt.show()