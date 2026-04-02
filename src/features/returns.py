import pandas as pd
import numpy as np


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola simple returns e log returns
    """
    df = df.copy()

    # simple return
    df["return"] = df["price"].pct_change()

    # log return
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))

    return df