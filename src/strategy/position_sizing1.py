import numpy as np


def compute_position_sizes(preds, df):
    """
    V3 Position Sizing Engine (FIXED LIVE + PROPORTIONAL SIZING)
    """

    preds = np.array(preds)

    # =========================
    # NORMALIZATION
    # =========================

    std = preds.std()

    if std == 0 or np.isnan(std):
        preds_z = np.zeros_like(preds)
    else:
        preds_z = (preds - preds.mean()) / std

    sizes = []

    trend = df["trend_200"].values[-len(preds):]
    vol = df["vol_20"].values[-len(preds):]

    vol_series = df["vol_20"]
    vol_mean = vol_series.rolling(252).mean()
    vol_regime = (vol_series / vol_mean).values[-len(preds):]

    # =========================
    # CORE LOOP
    # =========================

    for i in range(len(preds)):

        # =========================
        # LIVE FIX
        # =========================
        if len(preds) == 1:
            z = preds[i] * 10
        else:
            z = preds_z[i]

        # =========================
        # CORE SIZING (PROPORTIONAL)
        # =========================

        size = abs(z) * 0.6

        # penalizza segnali negativi (long-only)
        if z < 0:
            size *= 0.3

        # =========================
        # REGIME CONTROL
        # =========================

        if vol_regime[i] > 1.2:
            size *= 0.6
        elif vol_regime[i] < 0.8:
            size *= 1.1

        # =========================
        # TREND FILTER
        # =========================

        if trend[i] < 1:
            size *= 0.6

        # =========================
        # VOLATILITY TARGETING
        # =========================

        if vol[i] > 0:
            vol_scale = 0.018 / vol[i]
            vol_scale = min(vol_scale, 1.5)
            size *= vol_scale

        # =========================
        # EXTRA RISK CONTROL
        # =========================

        if z < -1:
            size *= 0.5

        # =========================
        # FINAL CLAMP
        # =========================

        size = max(0.0, min(size, 1.0))

        sizes.append(size)

    return np.array(sizes)