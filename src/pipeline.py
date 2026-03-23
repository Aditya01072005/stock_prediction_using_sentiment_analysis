import pandas as pd
import os


from news_fetcher import fetch_news
from sentiment_analysis import get_sentiment_score, get_sentiment_label


def run_pipeline(query="AAPL"):
    print("Running full pipeline...\n")

    news_df = fetch_news(query)

    print("News Fetched")
    print(news_df.head())


    news_df["Negative"] = news_df["Title"].apply(lambda x:get_sentiment_score(x)['neg'])
    news_df["Neutral"] = news_df["Title"].apply(lambda x:get_sentiment_score(x)['neu'])
    news_df["Positive"] = news_df["Title"].apply(lambda x:get_sentiment_score(x)['pos'])
    news_df["Compound"] = news_df["Title"].apply(lambda x:get_sentiment_score(x)['compound'])
    news_df["sentiment"] = news_df["Title"].apply(get_sentiment_label)

    print("After sentiment analysis")
    print(news_df.head())

    os.makedirs("data", exist_ok=True)

    file_path = "data/final_sentiment_data.csv"
    news_df.to_csv(file_path, index=False)

    print("File saved at:", file_path)

run_pipeline("AAPL")

