import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import load_model

print("Generating prediction plot...\n")


# LOAD MODEL + SCALER

model = load_model("model/lstm_model.h5", compile=False)
scaler = joblib.load("model/scaler.pkl")


# LOAD DATA

df = pd.read_csv("data/final_merged_data.csv")

features = ["Close", "RSI", "MACD", "Avg_Sentiment"]
df = df[features]

df.dropna(inplace=True)


# SCALE DATA

scaled_data = scaler.transform(df)


# CREATE SEQUENCES

X = []
y = []

sequence_length = 10

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i][0])

X, y = np.array(X), np.array(y)


# PREDICTIONS

predictions = model.predict(X)


# INVERSE SCALING

num_features = scaled_data.shape[1]

dummy_pred = np.zeros((len(predictions), num_features))
dummy_actual = np.zeros((len(y), num_features))

dummy_pred[:, 0] = predictions.flatten()
dummy_actual[:, 0] = y.flatten()

predicted_prices = scaler.inverse_transform(dummy_pred)[:, 0]
actual_prices = scaler.inverse_transform(dummy_actual)[:, 0]


# ALIGN PREDICTIONS

pred_plot = np.empty_like(actual_prices)
pred_plot[:] = np.nan

pred_plot[-len(predicted_prices):] = predicted_prices


# PLOT

plt.figure(figsize=(12,6))

plt.plot(actual_prices, label="Actual Price")
plt.plot(pred_plot, label="Predicted Price")

plt.title("Stock Price Prediction (LSTM + Sentiment)")
plt.xlabel("Time Steps")
plt.ylabel("Price")

plt.legend()
plt.grid(True)


plt.savefig("prediction_plot.png")

plt.show()

print("\n✅ Plot saved as prediction_plot.png")