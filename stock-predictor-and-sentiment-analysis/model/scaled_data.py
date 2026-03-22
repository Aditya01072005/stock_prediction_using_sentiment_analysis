from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

df = pd.read_csv("data/processed_stock_data.csv", index_col=0)

#dropping non-numeric columns
df = df.drop(columns=['Date', 'Stock'])

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

print(scaled_df.head())