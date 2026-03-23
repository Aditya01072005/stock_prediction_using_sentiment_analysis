import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

print("Running prediction...\n")

# Load model & scaler
model = load_model("model/lstm_model.h5", compile=False)
scaler = joblib.load("model/scaler.pkl")

# Load latest data
df = pd.read_csv("data/final_merged_data.csv")

# Select features
features = ["Close", "RSI", "MACD", "Avg_Sentiment"]
df = df[features]

# Drop NaN
df.dropna(inplace=True)

# Scale data
scaled_data = scaler.transform(df)

# Take last sequence
sequence_length = 10
last_sequence = scaled_data[-sequence_length:]

# Reshape for model
X_input = np.array([last_sequence])

# Predict
predicted_scaled = model.predict(X_input)

# Convert back to original scale
# Trick: create dummy array
dummy = np.zeros((1, scaled_data.shape[1]))
dummy[0][0] = predicted_scaled[0][0]  # Close is first column

predicted_price = scaler.inverse_transform(dummy)[0][0]

print("Predicted Next Day Close Price:", predicted_price)