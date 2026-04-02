import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """
    Carica un file CSV e ritorna un DataFrame pandas
    """
    df = pd.read_csv(path)
    return df


def basic_info(df: pd.DataFrame):
    """
    Stampa info base del dataset
    """
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nHead:")
    print(df.head())