"""data_cleaning.py
Utilities to load and clean the raw stock dataset.
"""
from pathlib import Path
import pandas as pd

def load_raw(path):
    path = Path(path)
    return pd.read_csv(path)

def clean_prices(df):
    # Basic cleaning placeholder: drop NaNs and convert dates
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    df = df.dropna()
    return df

def save_clean(df, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    raw = load_raw('../Dataset/all_stocks_5yr.csv')
    clean = clean_prices(raw)
    save_clean(clean, '../Dataset/cleaned_stocks_5yr.csv')
    print('Saved cleaned dataset to ../Dataset/cleaned_stocks_5yr.csv')
