import os
import pandas as pd
import numpy as np
from data_cleaning import clean_prices

def compute_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    # Determine the ticker column name (handle both 'ticker' and 'Name')
    ticker_col = 'ticker' if 'ticker' in df.columns else 'Name'
    
    # Sort by ticker and date
    df = df.sort_values([ticker_col, "date"]).copy()
    
    # Compute daily percentage returns per ticker
    df["return"] = df.groupby(ticker_col)["close"].pct_change()
    
    # Drop rows with NaN returns (first day for each ticker)
    df = df.dropna(subset=["return"])
    
    return df

def compute_mu_and_cov(df: pd.DataFrame) -> tuple:
    # Determine the ticker column name
    ticker_col = 'ticker' if 'ticker' in df.columns else 'Name'
    
    # Pivot to create a matrix: dates × tickers
    pivot = df.pivot(index="date", columns=ticker_col, values="return")
    
    # Keep only tickers with complete data (no NaN values)
    pivot = pivot.dropna(axis=1)
    
    # Compute mean returns (as numpy array)
    mu = pivot.mean().values
    
    # Compute covariance matrix (as numpy array)
    cov = pivot.cov().values
    
    # Get list of tickers
    tickers = pivot.columns.tolist()
    
    return mu, cov, tickers

def save_outputs(mu: np.ndarray, cov: np.ndarray, tickers: list, out_dir: str = "Results/processed") -> None:
    os.makedirs(out_dir, exist_ok=True)
    
    # Save mean returns as JSON
    import json
    mu_json_path = os.path.join(out_dir, "mean_returns.json")
    mu_dict = {ticker: float(mu[i]) for i, ticker in enumerate(tickers)}
    with open(mu_json_path, 'w') as f:
        json.dump(mu_dict, f, indent=2)
    print(f"[SUCCESS] Saved mean returns to {mu_json_path}")
    
    # Save covariance matrix as CSV with ticker labels
    cov_csv_path = os.path.join(out_dir, "cov_matrix.csv")
    pd.DataFrame(cov, index=tickers, columns=tickers).to_csv(cov_csv_path)
    print(f"[SUCCESS] Saved covariance matrix to {cov_csv_path}")
    
    # Save tickers as text file (one per line)
    tickers_path = os.path.join(out_dir, "tickers.txt")
    with open(tickers_path, 'w') as f:
        for ticker in tickers:
            f.write(f"{ticker}\n")
    print(f"[SUCCESS] Saved {len(tickers)} tickers to {tickers_path}")

def load_outputs(input_dir: str = "Results/processed") -> tuple:
    import json
    
    # Load mean returns from JSON
    with open(os.path.join(input_dir, "mean_returns.json"), 'r') as f:
        mu_dict = json.load(f)
    
    tickers = list(mu_dict.keys())
    mu = np.array([mu_dict[ticker] for ticker in tickers])
    
    # Load covariance matrix from CSV
    cov_df = pd.read_csv(os.path.join(input_dir, "cov_matrix.csv"), index_col=0)
    cov = cov_df.values
    
    return mu, cov, tickers

def main():
    """Main execution function."""
    print("=" * 50)
    print("COMPUTING RETURNS AND STATISTICS")
    print("=" * 50)
    
    data_path = "Dataset/all_stocks_5yr.csv"
    print(f"\nLoading data from {data_path}...")
    
    raw = pd.read_csv(data_path)
    raw["date"] = pd.to_datetime(raw["date"])
    print(f"Loaded {len(raw)} rows")

    # Clean
    print("\nCleaning data...")
    df = clean_prices(raw, verbose=False)
    print(f"After cleaning: {len(df)} rows, {df['ticker'].nunique()} unique tickers")

    # Prepare returns
    print("\nComputing daily returns...")
    df_returns = compute_daily_returns(df)
    print(f"Computed returns for {len(df_returns)} observations")
    
    print("\nComputing mean returns and covariance matrix...")
    mu, cov, tickers = compute_mu_and_cov(df_returns)
    print(f"Final portfolio: {len(tickers)} tickers")
    print(f"Mean return range: [{mu.min():.6f}, {mu.max():.6f}]")
    print(f"Covariance matrix shape: {cov.shape}")

    # Save results for other scripts
    print("\nSaving outputs...")
    save_outputs(mu, cov, tickers)
    
    # Display sample statistics
    print("\n--- SAMPLE STATISTICS ---")
    print(f"Top 5 tickers by mean return:")
    top_indices = np.argsort(mu)[-5:][::-1]
    for idx in top_indices:
        print(f"  {tickers[idx]}: {mu[idx]:.6f}")
    
    print(f"\nBottom 5 tickers by mean return:")
    bottom_indices = np.argsort(mu)[:5]
    for idx in bottom_indices:
        print(f"  {tickers[idx]}: {mu[idx]:.6f}")
    
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()