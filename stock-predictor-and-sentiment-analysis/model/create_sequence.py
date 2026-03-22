import pandas as pd
import numpy as np

df = pd.read_csv('data/processed_stock_data.csv')

cols = list(df.columns)
if 'Target' in cols:
    cols.remove('Target')
    cols.append('Target')
    df = df[cols]

df = df.drop(columns=['Date', 'Stock'])

#scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

data = scaled_data   #conerverting into numpy

#sequence func
def create_sequence(data, seq_length=60):
    x=[]
    y=[]

    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i, -1]) #target column

    return np.array(x), np.array(y)

x, y = create_sequence(data)

print("X shape: ", x.shape)
print("Y shape: ", y.shape)