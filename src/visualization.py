# src/visualization.py

import pandas as pd
import matplotlib.pyplot as plt


def plot_returns(df: pd.DataFrame):
    """
    Plot log returns.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df['log_return'])
    plt.title("NIFTY 50 Daily Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.tight_layout()
    plt.savefig("results/figures/log_returns.png")
    plt.show()


def plot_squared_returns(df: pd.DataFrame):
    """
    Plot squared returns as volatility proxy.
    """
    df['squared_return'] = df['log_return'] ** 2

    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df['squared_return'])
    plt.title("NIFTY 50 Squared Returns (Volatility Proxy)")
    plt.xlabel("Date")
    plt.ylabel("Squared Return")
    plt.tight_layout()
    plt.savefig("results/figures/squared_returns.png")
    plt.show()


if __name__ == "__main__":

    df = pd.read_csv(
        "data/processed/nifty_returns.csv",
        index_col=0,
        parse_dates=True
    )

    plot_returns(df)
    plot_squared_returns(df)