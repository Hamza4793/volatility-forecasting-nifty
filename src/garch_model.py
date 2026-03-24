# src/garch_model.py

import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train_test_split(series, train_ratio=0.8):
    split_index = int(len(series) * train_ratio)
    train = series[:split_index]
    test = series[split_index:]
    return train, test


def fit_garch(train_returns):
    model = arch_model(
        train_returns,
        vol="GARCH",
        p=1,
        q=1,
        mean="Zero",
        dist="normal"
    )
    result = model.fit(disp="off")
    return result


def evaluate_forecasts(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    return mse, mae, rmse


if __name__ == "__main__":

    df = pd.read_csv(
        "data/processed/nifty_returns.csv",
        index_col=0,
        parse_dates=True
    )

    returns = df["log_return"] * 100

    train_size = int(len(returns) * 0.8)

    train = returns[:train_size]
    test = returns[train_size:]

    predictions = []

    print("Starting rolling GARCH forecast...")

    for i in range(len(test)):
        
        train_data = returns[:train_size + i]

        model = arch_model(
            train_data,
            vol="GARCH",
            p=1,
            q=1,
            mean="Zero",
            dist="normal"
        )

        result = model.fit(disp="off")

        forecast = result.forecast(horizon=1)

        variance_forecast = forecast.variance.iloc[-1, 0]

        predictions.append(variance_forecast)

    actual_variance = (test ** 2).values

    mse = mean_squared_error(actual_variance, predictions)
    mae = mean_absolute_error(actual_variance, predictions)
    rmse = np.sqrt(mse)

    print("\nGARCH Rolling Out-of-Sample Performance:")
    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)