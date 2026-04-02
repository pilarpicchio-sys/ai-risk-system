import numpy as np
import pandas as pd
import yfinance as yf


assets = {
    "sp500": "^GSPC",
    "gold": "GC=F",
    "bonds": "TLT"
}


def build_dataset(ticker, name):

    df = yf.download(ticker, start="2000-01-01")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].rename(columns={"Close": "price"})
    df = df.reset_index()

    # returns
    df["return"] = df["price"].pct_change()
    df["log_return"] = np.log1p(df["return"])

    # features
    df["momentum_5"] = df["price"].pct_change(5)
    df["momentum_20"] = df["price"].pct_change(20)
    df["momentum_60"] = df["price"].pct_change(60)

    df["vol_20"] = df["return"].rolling(20).std()
    df["vol_60"] = df["return"].rolling(60).std()

    df["trend_50"] = df["price"] / df["price"].rolling(50).mean()
    df["trend_200"] = df["price"] / df["price"].rolling(200).mean()

    df["zscore_20"] = (
        df["price"] - df["price"].rolling(20).mean()
    ) / df["price"].rolling(20).std()

    # targets
    for h in [5, 21, 63]:
        df[f"target_{h}d"] = df["price"].pct_change(h).shift(-h)

    df = df.dropna().reset_index(drop=True)

    df.to_csv(f"data/{name}_multi_horizon.csv", index=False)

    print(f"{name} saved:", df.shape)


for name, ticker in assets.items():
    build_dataset(ticker, name)