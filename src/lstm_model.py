# src/lstm_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def create_sequences(data, sequence_length=5):
    X = []
    y = []

    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])

    return np.array(X), np.array(y)


if __name__ == "__main__":

    df = pd.read_csv(
        "data/processed/nifty_returns.csv",
        index_col=0,
        parse_dates=True
    )

    df["squared_return"] = (df["log_return"] * 100) ** 2

    values = df["squared_return"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    sequence_length = 5
    X, y = create_sequences(scaled_values, sequence_length)

    train_size = int(len(X) * 0.8)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_test = X[train_size:]
    y_test = y[train_size:]

    model = Sequential([
        LSTM(50, activation="tanh", input_shape=(sequence_length, 1)),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        verbose=1
    )

    predictions = model.predict(X_test)

    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test)

    mse = mean_squared_error(y_test_actual, predictions)
    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = np.sqrt(mse)

    print("\nLSTM Out-of-Sample Performance:")
    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)