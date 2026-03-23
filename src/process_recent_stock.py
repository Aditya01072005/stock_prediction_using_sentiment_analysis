import pandas as pd
import numpy as np
import os

print("Processing stock data...")

df = pd.read_csv("data/recent_stock_data.csv")

df["Date"] = pd.to_datetime(df["Date"]) #convert date column

# Fix numeric columns
numeric_cols = ["Open", "High", "Low", "Close", "Volume"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.sort_values("Date", inplace=True) # sort by dates

df["MA_7"] = df["Close"].rolling(window=7).mean()
df["MA_21"] = df["Close"].rolling(window=21).mean()


df["Return"] = df["Close"].pct_change()#Returns

df["Volatility"] = df["Return"].rolling(window=7).std() #volatility

df["Momentum"] = df["Close"] - df["Close"].shift(10)

#rsi
delta = df["Close"].diff()
gain = (delta.where(delta > 0,0)).rolling(window=14).mean()
loss = (delta.where(delta < 0,0)).rolling(window=14).mean()

rs = gain/loss
df["RSI"] = 100 - (100 / (1 + rs))

#MACD
exp1 = df["Close"].ewm(span=12, adjust=False).mean()
exp2 = df["Close"].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['Signal'] = df["MACD"].ewm(span=9, adjust=False).mean()

df['Target'] = df["Close"].shift(-1) #target (nest day price)

df.dropna(inplace=True) #dropping NaN values

df["Stock"] = "AAPL" #adding stock name

os.makedirs("data", exist_ok=True)
file_path = "data/processed_recent_stock_data.csv"
df.to_csv(file_path, index=False)

print("Processes stock data saved at:", file_path)
print("\nPreview:")
print(df.head())
