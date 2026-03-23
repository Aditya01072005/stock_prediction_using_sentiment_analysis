Stock Price Prediction with News Sentiment Analysis
Overview

This project integrates historical stock market data with news sentiment analysis to predict future stock prices using a deep learning model (LSTM).

The system combines:

Market indicators derived from stock prices
Sentiment signals extracted from financial news

The objective is to enhance prediction performance by incorporating real-world sentiment context alongside traditional time-series features.

Key Features
End-to-end data pipeline (stock + news → prediction)
News fetching using API
Sentiment analysis using VADER
Feature engineering (RSI, MACD, moving averages, volatility, momentum)
Time-series sequence generation
LSTM-based deep learning model
Model evaluation and visualization (Actual vs Predicted)
Next-day stock price prediction
Tech Stack
Python
Pandas, NumPy
Scikit-learn
TensorFlow / Keras
yFinance (stock data)
NewsAPI (news data)
VADER Sentiment Analysis
Project Structure
stock_prediction_using_sentiment_analysis/
│
├── src/
│   ├── fetch_stock_data.py
│   ├── process_recent_stock.py
│   ├── news_fetcher.py
│   ├── sentiment_analysis.py
│   ├── aggregate_sentiment.py
│   ├── merge_data.py
│   ├── train_model.py
│   ├── predict.py
│   ├── plot_predictions.py
│
├── data/                # Ignored in Git (generated datasets)
├── model/               # Saved model and scaler
│   ├── lstm_model.h5
│   ├── scaler.pkl
│
├── .env                 # API key (ignored)
├── .gitignore
├── requirements.txt
├── README.md
Workflow
Fetch recent stock data using yFinance
Perform feature engineering (RSI, MACD, moving averages, etc.)
Fetch financial news using NewsAPI
Apply sentiment analysis using VADER
Aggregate sentiment scores by date
Merge stock data with sentiment data
Scale features and generate sequences
Train LSTM model
Evaluate model performance
Predict next-day stock price
Features Used
Close Price
RSI (Relative Strength Index)
MACD (Moving Average Convergence Divergence)
Moving Averages (MA7, MA21)
Volatility and Momentum
News Sentiment Score (Aggregated)
Model Details
Model: LSTM (Long Short-Term Memory)
Input: Time-series sequences (window size = 10)
Output: Next-day closing price
Loss Function: Mean Squared Error (MSE)
Optimizer: Adam
How to Run
# Clone repository
git clone https://github.com/Aditya01072005/stock_prediction_using_sentiment_analysis.git

cd stock_prediction_using_sentiment_analysis

# Install dependencies
pip install -r requirements.txt

# Step 1: Fetch stock data
python src/fetch_stock_data.py

# Step 2: Process stock data
python src/process_recent_stock.py

# Step 3: Fetch news
python src/news_fetcher.py

# Step 4: Sentiment analysis
python src/pipeline.py

# Step 5: Aggregate sentiment
python src/aggregate_sentiment.py

# Step 6: Merge datasets
python src/merge_data.py

# Step 7: Train model
python src/train_model.py

# Step 8: Predict
python src/predict.py

# Step 9: Visualization
python src/plot_predictions.py
Output
Trained model: model/lstm_model.h5
Scaler: model/scaler.pkl
Prediction plot: prediction_plot.png
Predicted next-day stock price (console output)
Future Improvements
Multi-stock training support
Advanced NLP models (BERT / FinBERT)
Real-time prediction pipeline
Deployment using Streamlit or Flask
Hyperparameter tuning for improved accuracy
Author

Aditya Chauhan

Notes
API keys are stored securely using environment variables and are not included in this repository.
Generated datasets are excluded via .gitignore.