import yfinance as yf
import pandas as pd
import os

print("Fetching stock data...")

df = yf.download("AAPL", period="1y")

df.reset_index(inplace=True)

print(df.head())

os.makedirs("data", exist_ok=True)

file_path = "data/recent_stock_data.csv"
df.to_csv(file_path, index=False)

print("Stock data saved successfully at: ", file_path)