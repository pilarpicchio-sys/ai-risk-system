import pandas as pd
import numpy as np


def compute_advanced_features(df):

    # ===== TREND =====
    df["ma_10"] = df["price"].rolling(10).mean()
    df["ma_50"] = df["price"].rolling(50).mean()

    df["trend"] = df["ma_10"] / df["ma_50"] - 1
    

    # ==== MOMENTUM ====

    df["momentum_10"] = df["price"].pct_change(10)


    # ===== Z-SCORE =====
    rolling_mean = df["price"].rolling(20).mean()
    rolling_std = df["price"].rolling(20).std()

    df["zscore"] = (df["price"] - rolling_mean) / rolling_std


    # ===== VOLATILITY REGIME =====
    df["vol_20"] = df["return"].rolling(20).std()
    df["vol_regime"] = df["vol_20"] / df["vol_20"].rolling(100).mean()


    # ===== DRAWDOWN =====
    cummax = df["price"].cummax()
    df["drawdown"] = (df["price"] - cummax) / cummax


    return df