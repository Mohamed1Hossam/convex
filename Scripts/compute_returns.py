# Scripts/compute_returns.py
import os
import pandas as pd
import numpy as np
from data_cleaning import clean_prices

def compute_daily_returns(df):
    """Compute daily percentage returns for each ticker."""
    df = df.sort_values(["Name", "date"])
    df["return"] = df.groupby("Name")["close"].pct_change()
    df = df.dropna(subset=["return"])
    return df

def compute_mu_and_cov(df):
    """Compute mean returns vector μ and covariance matrix Σ."""
    pivot = df.pivot(index="date", columns="Name", values="return")
    pivot = pivot.dropna(axis=1)  # keep only complete tickers

    mu = pivot.mean().to_dict()
    cov = pivot.cov()

    return mu, cov, pivot.columns.tolist()

def save_outputs(mu, cov, tickers):
    """Save processed results into Results/processed/"""
    out_dir = "Results/processed"
    os.makedirs(out_dir, exist_ok=True)

    # Save mean returns
    pd.Series(mu).to_json(os.path.join(out_dir, "mean_returns.json"))

    # Save covariance matrix
    cov.to_csv(os.path.join(out_dir, "cov_matrix.csv"))

    # Save the final list of included tickers (no header)
    pd.Series(tickers).to_csv(os.path.join(out_dir, "selected_tickers.csv"),
                              index=False, header=False)

if __name__ == "__main__":
    data_path = "Dataset/all_stocks_5yr.csv"
    raw = pd.read_csv(data_path)
    raw["date"] = pd.to_datetime(raw["date"])

    # Clean
    df = clean_prices(raw)

    # Prepare returns
    df_returns = compute_daily_returns(df)
    mu, cov, tickers = compute_mu_and_cov(df_returns)

    # Save results for other scripts
    save_outputs(mu, cov, tickers)

    print("Daily returns, mean vector, and covariance saved in Results/processed/")