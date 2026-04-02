import pandas as pd


def prepare_dataset(df: pd.DataFrame, target_col: str):
    """
    Prepara dataset per training
    """
    df = df.copy()

    # scegli SOLO colonne utili (niente horizon lunghi)
    feature_cols = [
        "return",
        "log_return",
        "rolling_mean_2",
        "rolling_vol_2",
        "momentum_2",
    ]

    # tieni solo colonne necessarie
    df = df[feature_cols + [target_col]]

    # rimuovi NaN SOLO su queste
    df = df.dropna()

    X = df[feature_cols]
    y = df[target_col]

    return X, y