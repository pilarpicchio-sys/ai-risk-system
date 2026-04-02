import numpy as np
import pandas as pd
import yfinance as yf

from src.data.load_data import load_csv


# =========================
# LOAD GOLD (TUO FILE)
# =========================

gold = load_csv("data/gold.csv")
gold = gold.rename(columns={"price": "gold_price"})

gold["date"] = pd.to_datetime(gold["date"])
gold = gold.set_index("date")

gold["gold_ret"] = gold["gold_price"].pct_change()


# =========================
# DOWNLOAD MACRO DATA
# =========================

tickers = {
    "sp500": "^GSPC",
    "usd": "DX-Y.NYB",
    "rates": "^TNX"
}

data = {}

for name, ticker in tickers.items():
    df = yf.download(ticker, start=gold.index.min(), end=gold.index.max())
    df = df[["Close"]].rename(columns={"Close": f"{name}_price"})
    data[name] = df


# =========================
# MERGE
# =========================

df = gold.copy()

for name in data:
    df = df.join(data[name], how="left")


# =========================
# RETURNS
# =========================

df["sp500_ret"] = df["sp500_price"].pct_change()
df["usd_ret"] = df["usd_price"].pct_change()
df["rate_ret"] = df["rates_price"].pct_change()


# =========================
# FEATURES
# =========================

df["sp500_mom_20"] = df["sp500_price"].pct_change(20)
df["usd_mom_20"] = df["usd_price"].pct_change(20)
df["rate_mom_20"] = df["rates_price"].pct_change(20)

df["sp500_vol_20"] = df["sp500_ret"].rolling(20).std()
df["usd_vol_20"] = df["usd_ret"].rolling(20).std()

df["gold_vs_sp500"] = df["gold_ret"] - df["sp500_ret"]
df["gold_vs_usd"] = df["gold_ret"] - df["usd_ret"]


# =========================
# TARGET
# =========================

h = 21
df["target"] = df["gold_price"].pct_change(h).shift(-h)


# =========================
# CLEAN
# =========================

df = df.dropna()


# =========================
# CHECK
# =========================

print("\nDataset shape:", df.shape)

corr = df.corr(numeric_only=True)["target"].sort_values(ascending=False)

print("\nTop correlations:")
print(corr.head(10))

print("\nWorst correlations:")
print(corr.tail(10))


# =========================
# SAVE
# =========================

df.to_csv("data/macro_dataset.csv")