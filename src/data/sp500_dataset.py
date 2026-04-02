import numpy as np
import pandas as pd
import yfinance as yf


# =========================
# DOWNLOAD SP500
# =========================

df = yf.download("^GSPC", start="2000-01-01")

# fix colonne
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[["Close"]].rename(columns={"Close": "price"})

df.index.name = "date"
df = df.reset_index()


# =========================
# RETURNS
# =========================

df["return"] = df["price"].pct_change()
df["log_return"] = np.log1p(df["return"])


# =========================
# FEATURES (BASE)
# =========================

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


# =========================
# TARGET (21d RETURN)
# =========================

h = 21
df["target"] = df["price"].pct_change(h).shift(-h)


# =========================
# CLEAN
# =========================

df = df.dropna().reset_index(drop=True)


# =========================
# SAVE
# =========================

df.to_csv("data/sp500_dataset.csv", index=False)

print("Dataset ready:", df.shape)
print(df.head())