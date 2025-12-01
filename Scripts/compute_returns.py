import pandas as pd
import numpy as np
import os

def compute_daily_returns(df):
    """Compute daily percentage returns for each ticker."""
    df = df.sort_values(["Name", "date"])
    df["return"] = df.groupby("Name")["close"].pct_change()
    df = df.dropna(subset=["return"])
    return df

def compute_mu_and_cov(df):
    """Compute mean returns vector μ and covariance matrix Σ."""
    pivot = df.pivot(index="date", columns="Name", values="return")
    pivot = pivot.dropna(axis=1)     # keep only complete tickers

    mu = pivot.mean().to_dict()
    cov = pivot.cov()

    return mu, cov, pivot.columns.tolist()

def save_outputs(mu, cov, tickers):
    """Save processed results into Results/processed/"""
    os.makedirs("../Results/processed", exist_ok=True)

    # Save mean returns
    pd.Series(mu).to_json("../Results/processed/mean_returns.json")

    # Save covariance matrix
    cov.to_csv("../Results/processed/cov_matrix.csv")

    # Save the final list of included tickers
    pd.Series(tickers).to_csv("../Results/processed/selected_tickers.csv", index=False)

if __name__ == "__main__":
    data_path = "../Dataset/all_stocks_5yr.csv"

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    # Prepare returns
    df_returns = compute_daily_returns(df)
    mu, cov, tickers = compute_mu_and_cov(df_returns)

    # Save results for other scripts
    save_outputs(mu, cov, tickers)

    print("Daily returns, mean vector , and covariance saved in Results/processed/")
