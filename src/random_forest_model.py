# src/random_forest_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def create_lag_features(series, lags=5):
    df = pd.DataFrame()
    df["target"] = series

    for i in range(1, lags + 1):
        df[f"lag_{i}"] = series.shift(i)

    df = df.dropna()
    return df


if __name__ == "__main__":

    df = pd.read_csv(
        "data/processed/nifty_returns.csv",
        index_col=0,
        parse_dates=True
    )

    # Target = squared returns
    df["squared_return"] = (df["log_return"] * 100) ** 2

    data = create_lag_features(df["squared_return"], lags=5)

    train_size = int(len(data) * 0.8)

    train = data[:train_size]
    test = data[train_size:]

    X_train = train.drop("target", axis=1)
    y_train = train["target"]

    X_test = test.drop("target", axis=1)
    y_test = test["target"]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print("\nRandom Forest Out-of-Sample Performance:")
    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)