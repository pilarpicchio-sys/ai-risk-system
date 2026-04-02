import yfinance as yf
import pandas as pd


def download_gold_data():
    """
    Scarica dati oro (ETF GLD)
    """
    data = yf.download("GLD", start="2010-01-01")

    df = data.reset_index()

    # teniamo solo quello che serve
    df = df[["Date", "Close"]]
    df.columns = ["date", "price"]

    return df


if __name__ == "__main__":
    df = download_gold_data()
    df.to_csv("data/gold.csv", index=False)
    print("Saved data/gold.csv")