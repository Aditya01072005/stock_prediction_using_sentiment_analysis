import yfinance as yf
import pandas as pd

stocks = [
"RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
"SBIN.NS","ITC.NS","LT.NS","AXISBANK.NS","BAJFINANCE.NS",
"KOTAKBANK.NS","HINDUNILVR.NS","BHARTIARTL.NS","ASIANPAINT.NS",
"MARUTI.NS","TITAN.NS","ULTRACEMCO.NS","SUNPHARMA.NS","WIPRO.NS",
"AAPL","MSFT","AMZN","GOOGL","TSLA","NVDA","META","NFLX","INTC",
"AMD","IBM","ORCL","KO","PEP","NKE","MCD","DIS","PYPL","V","MA"
]

data = yf.download(stocks, start="2010-01-01", end="2025-01-01")

#print(data.head())


data.to_csv("stock_data.csv")

print("stock data saved successfully!")