import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

st.title("Stock Price Prediction with Sentiment Analysis")

model = load_model("model/lstm_model.h5", compile=False)
scaler = joblib.load("model/scaler.pkl")

# Load data
df = pd.read_csv("data/final_merged_data.csv")

features = ["Close", "RSI", "MACD", "Avg_Sentiment"]
df = df[features]
df.dropna(inplace=True)

st.subheader("Latest Data")
st.dataframe(df.tail())

# Scale
scaled_data = scaler.transform(df)

# Create sequence
sequence_length = 10
last_sequence = scaled_data[-sequence_length:]

X_input = np.array([last_sequence])

# Predict
predicted_scaled = model.predict(X_input)

# Inverse scaling
dummy = np.zeros((1, scaled_data.shape[1]))
dummy[0][0] = predicted_scaled[0][0]

predicted_price = scaler.inverse_transform(dummy)[0][0]

st.subheader("Predicted Next Day Price")
st.success(f"{predicted_price:.2f}")

# Plot
st.subheader("Price Trend")
st.line_chart(df["Close"])