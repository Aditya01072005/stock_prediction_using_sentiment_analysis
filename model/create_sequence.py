import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



df = pd.read_csv('data/processed_stock_data.csv')

stock_name = df['Stock'].unique()[0]
df = df[df['Stock'] == stock_name]

print("Using stock:", stock_name)

cols = list(df.columns)
if 'Target' in cols:
    cols.remove('Target')
    cols.append('Target')
    df = df[cols]

df = df.drop(columns=['Date', 'Stock'])

train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

#scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.fit_transform(test_df)

joblib.dump(scaler, "model/scaler.pkl")


#sequence func
def create_sequence(data, seq_length=60):
    x=[]
    y=[]

    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i, -1]) #target column

    return np.array(x), np.array(y)

x_train, y_train = create_sequence(train_scaled)
x_test, y_test = create_sequence(test_scaled)

print("X train: ", x_train.shape)
print("X test: ", x_test.shape)