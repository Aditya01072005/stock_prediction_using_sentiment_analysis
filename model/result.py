import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

from create_sequence import x_test, y_test

# ==============================
# LOAD MODEL + SCALER
# ==============================
model = load_model("model/stock_lstm_model.h5")
scaler = joblib.load("model/scaler.pkl")

# ==============================
# LOAD ORIGINAL DATA
# ==============================
df = pd.read_csv("data/processed_stock_data.csv")

# 🔥 IMPORTANT: USE ONLY ONE STOCK
df = df[df['Stock'] == df['Stock'].unique()[0]]  # pick first stock

# Move Target to last (same as training)
cols = list(df.columns)
if 'Target' in cols:
    cols.remove('Target')
    cols.append('Target')
    df = df[cols]

# Drop non-numeric
df = df.drop(columns=['Date', 'Stock'])

# ==============================
# PREDICT
# ==============================
predictions = model.predict(x_test)

# ==============================
# INVERSE SCALING
# ==============================
num_features = x_test.shape[2]

dummy_pred = np.zeros((len(predictions), num_features))
dummy_actual = np.zeros((len(y_test), num_features))

dummy_pred[:, -1] = predictions.flatten()
dummy_actual[:, -1] = y_test.flatten()

predicted_prices = scaler.inverse_transform(dummy_pred)[:, -1]
actual_prices = scaler.inverse_transform(dummy_actual)[:, -1]

# ==============================
# FIX ALIGNMENT
# ==============================
pred_plot = np.empty_like(actual_prices)
pred_plot[:] = np.nan

# place predictions at correct position
pred_plot[-len(predicted_prices):] = predicted_prices

# ==============================
# PLOT
# ==============================
plt.figure(figsize=(12,6))

plt.plot(actual_prices, label="Actual Price")
plt.plot(pred_plot, label="Predicted Price")

plt.title("Actual vs Predicted Stock Prices (LSTM)")
plt.style.use('seaborn-v0_8-darkgrid')
plt.tight_layout()
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

# Save image for LinkedIn
plt.savefig("prediction_plot.png")

plt.show()