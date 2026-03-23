from newsapi import NewsApiClient
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")

if not API_KEY:
    raise ValueError("API key not found")

newsapi = NewsApiClient(api_key=API_KEY)

def fetch_news(query="AAPL"):
    articles = newsapi.get_everything(
        q = query,
        language='en',
        sort_by='publishedAt',
        page_size=40
    )
    
    data = []

    for article in articles['articles']:
        data.append({
            "Date": article['publishedAt'][:10],
            "Title" : article['title']
        })

    return pd.DataFrame(data)

print("Fetching news")

df = fetch_news("AAPL")

print(df.head())

os.makedirs("data", exist_ok=True)

file_path = "data/news_data.csv"
df.to_csv(file_path, index=False)

print("News data saved at:", file_path)



