import pandas as pd


def compute_multi_horizon_returns(df: pd.DataFrame, horizons=None) -> pd.DataFrame:
    if horizons is None:
        horizons = [5, 21, 63]

    df = df.copy()

    for h in horizons:
        df[f"return_{h}d"] = df["price"].pct_change(periods=h)

    return df