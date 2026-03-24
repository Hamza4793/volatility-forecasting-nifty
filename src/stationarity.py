# src/stationarity.py

import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_test(series, title=""):
    """
    Perform ADF test and print results.
    """

    print(f"Augmented Dickey-Fuller Test: {title}")
    result = adfuller(series)

    labels = [
        "ADF Test Statistic",
        "p-value",
        "Lags Used",
        "Number of Observations"
    ]

    for value, label in zip(result[:4], labels):
        print(f"{label}: {value}")

    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value}")

    if result[1] <= 0.05:
        print("Conclusion: Series is STATIONARY (reject H0)")
    else:
        print("Conclusion: Series is NON-STATIONARY (fail to reject H0)")


if __name__ == "__main__":

    df = pd.read_csv(
        "data/processed/nifty_returns.csv",
        index_col=0,
        parse_dates=True
    )

    adf_test(df["log_return"], title="NIFTY Log Returns")