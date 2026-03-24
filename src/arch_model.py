# src/arch_model.py

import pandas as pd
from arch import arch_model


def fit_arch_model(returns):

    model = arch_model(
        returns,
        vol="ARCH",
        p=1,
        mean="Zero",     # assume zero mean
        dist="normal"
    )

    result = model.fit(disp="off")

    return result


if __name__ == "__main__":

    df = pd.read_csv(
        "data/processed/nifty_returns.csv",
        index_col=0,
        parse_dates=True
    )

    returns = df["log_return"] * 100  # scale for stability

    result = fit_arch_model(returns)

    print(result.summary())