# Stock Price Prediction with News Sentiment Analysis

## Overview

This project combines **stock market data** and **news sentiment analysis** to predict future stock prices using Machine Learning.

It leverages:

* Recent stock data using yFinance 
* News sentiment analysis 
* Deep learning models (LSTM) 

The goal is to improve prediction accuracy by incorporating **real-world sentiment signals** along with price trends.

---

## Key Features

* Stock price prediction using LSTM
* Sentiment analysis on financial news
* Integration of sentiment + stock data
* Time-series sequence generation
* Data preprocessing pipeline
* Visualization of predictions
* News fetching using API

---

## Tech Stack

* Python 
* Pandas & NumPy
* Scikit-learn
* TensorFlow / Keras
* NLP for sentiment analysis
* yFinance (stock data)
* NewsApi (news data)
* NLP (VADER Sentiment Analysis)

---

## Workflow

1. Fetch recent stock price data using yFinance 
2. Clean and preprocess data
3. Perform feature engineering (RSI, MACD, etc.)
4. Fetch financial news using NewsAPI 
5. Perform sentiment analysis on news
6. Aggregate sentiment scores by date
7. Merge sentiment data with stock data
8. Scale data and generate sequences for LSTM
9. Train the model
10. Predict future stock prices
11. Visualize predictions 

---

## Features Used

* Closing Price
* Moving Averages(MA7, MA21)
* Sentiment Score
* Time-series sequences
* RSI (Relative Strength Index)
* MACD (Moving Average Convergence Divergence)


---

## Model

* LSTM (Long Short-Term Memory)
* Suitable for time-series forecasting
* Captures temporal dependencies in stock data
* Uses both market indicators and sentiment signals

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/Aditya01072005/stock_prediction_using_sentiment_analysis.git

# Navigate into project folder
cd stock_prediction_using_sentiment_analysis

# Install dependencies
pip install -r requirements.txt

# Run step-by-step pipeline

# Fetch stock data
python src/fetch_stock_data.py

# Process stock data
python src/process_recent_stock.py

# Fetch news data
python src/news_fetcher.py

# Run sentiment pipeline
python src/pipeline.py

# Aggregate sentiment
python src/aggregate_sentiment.py

# Merge stock and sentiment data
python src/merge_data.py

# Train model
python src/train_model.py

# Predict next-day price
python src/predict.py

# Visualize predictions
python src/plot_predictions.py
```

---

## Future Improvements

* Multi-stock training support
* Advanced NLP models (BERT / FinBERT)
* Real-time prediction system
* Web app deployment

---

## Author

Aditya Chauhan

---

## Support

If you like this project, consider giving it a ⭐ on GitHub!
