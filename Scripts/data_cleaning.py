# Scripts/data_cleaning.py
import os
import pandas as pd

REQUIRED_COLS = ["date", "Name", "open", "high", "low", "close", "volume"]

def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in REQUIRED_COLS if c in df.columns]
    df = df[cols].copy()

    # Types
    df["date"] = pd.to_datetime(df["date"])
    df["Name"] = df["Name"].astype(str)

    # Basic cleaning
    df = df.drop_duplicates(subset=["Name", "date"])
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    # Remove non-positive values
    for c in ["open", "high", "low", "close", "volume"]:
        df = df[df[c] > 0]

    df = df.sort_values(["Name", "date"])
    return df

def main():
    os.makedirs("../Results/processed", exist_ok=True)
    src = "../Data/all_stocks_5yr.csv"
    df = pd.read_csv(src)
    df = clean_prices(df)
    df.to_csv("../Results/processed/clean_prices.csv", index=False)
    print("Saved cleaned prices to ../Results/processed/clean_prices.csv")

if __name__ == "__main__":
    main()