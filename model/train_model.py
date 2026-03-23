import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


print("Model Started...")

df = pd.read_csv("data/final_merged_data.csv")

features = ["Close", "RSI", "MACD", "Avg_Sentiment"]
df = df[features]

df.dropna(inplace=True)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)


x = []
y = []

sequence_length = 10

for i in range(sequence_length, len(scaled_data)):
    x.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i][0])

x, y = np.array(x), np.array(y)

print("Data Shapes: ")
print("X:", x.shape)
print("Y:", y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=False
)

print("\nTrain/Test Split:")
print("Train:", x_train.shape)
print("Test:", x_test.shape)



model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(64))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer = "adam", loss = "mse")

print("\nTrainning Started\n")

model.fit(x_train,y_train, epochs=10, batch_size=16)



loss = model.evaluate(x_test, y_test)
print("\n Test loss: ", loss)

os.makedirs("model", exist_ok=True)

model.save("model/lstm_model.h5")
joblib.dump(scaler, "model/scaler.pkl")

print("Model and scaler are saved")