# src/data_loader.py

import yfinance as yf
import pandas as pd
from datetime import datetime


def download_nifty_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads clean NIFTY 50 data.
    """

    ticker = "^NSEI"

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,   # Adjust prices automatically
        progress=False
    )

    # Remove multi-index if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


if __name__ == "__main__":

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = "2014-01-01"

    df = download_nifty_data(start_date, end_date)

    print(df.head())
    print("\nColumn Types:\n", df.dtypes)

    df.to_csv("data/raw/nifty_raw.csv")