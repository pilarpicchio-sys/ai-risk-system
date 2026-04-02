import pandas as pd


def create_buckets(df: pd.DataFrame, column: str, bins=None) -> pd.DataFrame:
    """
    Trasforma returns in bucket discreti
    """
    if bins is None:
        bins = [-float("inf"), -0.02, -0.01, 0, 0.01, 0.02, float("inf")]

    labels = list(range(len(bins) - 1))

    df = df.copy()
    df[f"{column}_bucket"] = pd.cut(df[column], bins=bins, labels=labels)

    return df
    
def create_multi_horizon_buckets(df: pd.DataFrame, horizons=None) -> pd.DataFrame:
    """
    Crea bucket per tutti gli orizzonti
    """
    if horizons is None:
        horizons = [5, 21, 63]

    df = df.copy()

    for h in horizons:
        col = f"return_{h}d"
        if col in df.columns:
            df = create_buckets(df, col)

    return df