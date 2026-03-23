import pandas as pd
import os

print("Starting merge process...\n")

# Load processed stock data
stock_df = pd.read_csv("data/processed_recent_stock_data.csv")

# Load sentiment data
sentiment_df = pd.read_csv("data/daily_sentiment.csv")

# Convert Date columns to datetime
stock_df["Date"] = pd.to_datetime(stock_df["Date"])
sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"])

# IMPORTANT: Filter stock data to recent dates (to match news)
stock_df = stock_df[stock_df["Date"] >= "2026-01-01"]

print("Stock data after filtering:")
print(stock_df.head())

print("\nSentiment data:")
print(sentiment_df.head())

# Merge datasets
merged_df = pd.merge(stock_df, sentiment_df, on="Date", how="left")

# Handle missing sentiment values
if "Compound" in merged_df.columns:
    merged_df["Compound"] = merged_df["Compound"].fillna(0)
    merged_df.rename(columns={"Compound": "Avg_Sentiment"}, inplace=True)

elif "Avg_Sentiment" in merged_df.columns:
    merged_df["Avg_Sentiment"] = merged_df["Avg_Sentiment"].fillna(0)

else:
    raise ValueError("Sentiment column not found in sentiment file")

# Drop unwanted column if exists
if "Unnamed: 0" in merged_df.columns:
    merged_df.drop(columns=["Unnamed: 0"], inplace=True)

# Final preview
print("\nFinal Merged Data:")
print(merged_df.head())

# Create data folder
os.makedirs("data", exist_ok=True)

# Save final dataset
file_path = "data/final_merged_data.csv"
merged_df.to_csv(file_path, index=False)

print("\nFinal merged dataset saved at:", file_path)