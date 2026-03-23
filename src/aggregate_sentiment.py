import pandas as pd

df = pd.read_csv("data/final_sentiment_data.csv")

df["Date"] = pd.to_datetime(df["Date"])


daily_sentiment = df.groupby("Date")["Compound"].mean().reset_index()


daily_sentiment.rename(columns={"Compund" : "Avg_Sentiment"}, inplace=True)

print(daily_sentiment.head())


daily_sentiment.to_csv("data/daily_sentiment.csv", index=False)

print("Daily Sentiment saved!")