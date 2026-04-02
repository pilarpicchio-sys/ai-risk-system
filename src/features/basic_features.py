import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature base per il modello
    """
    df = df.copy()

    # returns già presenti
    if "return" in df.columns:

        # rolling mean
        df["rolling_mean_2"] = df["return"].rolling(2).mean()

        # rolling volatility
        df["rolling_vol_2"] = df["return"].rolling(2).std()

        # momentum semplice
        df["momentum_2"] = df["price"] / df["price"].shift(2) - 1

    return df