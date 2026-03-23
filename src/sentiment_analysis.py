from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import os

print("Script started")

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    return analyzer.polarity_scores(text)

def get_sentiment_label(text):
    score = analyzer.polarity_scores(text)
    compound = score['compound']

    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


print("Started")

news_list = [
    "Reliance stock rises after a strong earnings", 
    "Market crashes due to global recession fears", 
    "Infosys shows stable growth in Q3",
    "Tech stocks fall sharply amid uncertainty", 
    "Banking sector sees positive momentum"
]

data =[]

for news in news_list:
    score = get_sentiment_score(news)
    label = get_sentiment_label(news)

    print(f"News: {news}")
    print(f"Score: {score}")
    print(f"Label: {label}")
    print("-" * 50)

    data.append({
        "Date": "2024-03-21",
        "News" : news,
        "Negative" :score['neg'],
        "Neutral" : score['neu'],
        "Positive" : score['pos'],
        "Compound" : score['compound'],
        "Sentiment" : label
    })


df = pd.DataFrame(data)

os.makedirs("data", exist_ok=True)

file_path = "data/sentiment_data.csv"
df.to_csv(file_path, index=False)

print("\n Data saved successfully at:", file_path)
print("\n Final Dataset:")
print(df)

