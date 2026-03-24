# src/preprocessing.py

import pandas as pd
import numpy as np


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily log returns from closing prices.
    """
    
    df = df.copy()

    # Ensure Close column is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    # Compute log returns
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Drop missing values
    df = df.dropna()

    return df


if __name__ == "__main__":
    
    df = pd.read_csv(
        "data/raw/nifty_raw.csv",
        index_col=0
    )

    # Convert index to datetime explicitly
    df.index = pd.to_datetime(df.index)

    # Print data types (for debugging)
    print("Column Types:\n", df.dtypes)

    df = compute_log_returns(df)

    print(df[['Close', 'log_return']].head())

    df.to_csv("data/processed/nifty_returns.csv")