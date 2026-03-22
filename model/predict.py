import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("model/stock_lstm_model.h5")
scaler = joblib.load("model/scaler.pkl")

df = pd.read_csv("data/processed_stock_data.csv")

cols = list(df.columns)
if 'Target' in cols:
    cols.remove('Target')
    cols.append('Target')
    df = df[cols]


df = df.drop(columns=['Date', 'Stock'])


scaled_data = scaler.transform(df)


SEQ_LENGTH = 60

last_sequence = scaled_data[-SEQ_LENGTH:]

x_test = []
x_test.append(last_sequence)

x_test = np.array(x_test)


prediction = model.predict(x_test)



dummy = np.zeros((1, scaled_data.shape[1]))
dummy[0,-1] = prediction[0][0]

predicted_price = scaler.inverse_transform(dummy)[0,-1]

print("Predicted Next Value: ", predicted_price)
