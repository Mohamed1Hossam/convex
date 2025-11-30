"""compute_returns.py
Compute simple and log returns from price data.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def compute_log_returns(price_df):
    # expects price_df with columns: date, ticker, close or wide-format prices
    df = price_df.copy()
    # If tidy long format with 'close' column
    if 'close' in df.columns and 'ticker' in df.columns:
        df = df.pivot(index='date', columns='ticker', values='close')
    returns = np.log(df / df.shift(1)).dropna()
    return returns

if __name__ == '__main__':
    p = Path('../Dataset/cleaned_stocks_5yr.csv')
    if p.exists():
        df = pd.read_csv(p, parse_dates=['date'])
        r = compute_log_returns(df)
        r.to_csv('../Dataset/returns_5yr.csv')
        print('Saved returns to ../Dataset/returns_5yr.csv')
    else:
        print('Cleaned dataset not found; run data_cleaning first')
