import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


from create_sequence import x,y

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, shuffle=False
    )

print("Train Shape: ", x_train.shape)
print("Test Shape: ", x_test.shape)

#LSTM Model

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(1))


model.compile(
    optimizer = 'adam', 
    loss = 'mean_squared_error'
)

history = model.fit(
    x_train,
    y_train,
    epochs = 10, 
    batch_size = 32, 
    validation_data=(x_test, y_test)
)


loss = model.evaluate(x_test,y_test)
print("Test loss: ", loss)

model.save("model/stock_lstm_model.h5")
print("Model saved successfully")