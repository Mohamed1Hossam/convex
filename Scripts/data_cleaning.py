# Scripts/data_cleaning.py
import os
import pandas as pd

REQUIRED_COLS = ["date", "Name", "open", "high", "low", "close", "volume"]

def audit_data_quality(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Perform comprehensive data quality checks.
    Returns a dictionary with audit results.
    """
    results = {}
    
    # 1. Check for Low > High (The "Impossible" Price)
    bad_logic = df[df['low'] > df['high']]
    results['low_gt_high'] = len(bad_logic)
    
    if verbose:
        print(f"1. Rows where Low > High: {len(bad_logic)}")
        if not bad_logic.empty:
            print("   Sample of bad logic:")
            print(bad_logic[['date', 'Name', 'high', 'low']].head())
        else:
            print("   [OK] Logic check passed.")
        print("-" * 30)
    
    # 2. Check for Zero or Negative Volume
    bad_volume = df[df['volume'] <= 0]
    results['bad_volume'] = len(bad_volume)
    
    if verbose:
        print(f"2. Rows with Volume <= 0: {len(bad_volume)}")
        if not bad_volume.empty:
            print("   Sample of bad volume:")
            print(bad_volume[['date', 'Name', 'volume']].head())
        else:
            print("   [OK] Volume check passed.")
        print("-" * 30)
    
    # 3. Check for Zero or Negative Price (using Close as reference)
    bad_price = df[df['close'] <= 0]
    results['bad_price'] = len(bad_price)
    
    if verbose:
        print(f"3. Rows with Price <= 0: {len(bad_price)}")
        if not bad_price.empty:
            print("   Sample of bad prices:")
            print(bad_price[['date', 'Name', 'close']].head())
        else:
            print("   [OK] Price check passed.")
        print("-" * 30)
    
    return results

def clean_prices(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Enhanced cleaning function with comprehensive validation.
    
    Args:
        df: Raw DataFrame with stock price data
        verbose: Whether to print detailed cleaning reports
        
    Returns:
        Cleaned DataFrame
    """
    if verbose:
        print(f"Initial rows: {len(df)}")
    
    # Select required columns
    cols = [c for c in REQUIRED_COLS if c in df.columns]
    df = df[cols].copy()

    # Convert types
    df["date"] = pd.to_datetime(df["date"])
    df["Name"] = df["Name"].astype(str)

    # Remove duplicates
    before_dup = len(df)
    df = df.drop_duplicates(subset=["Name", "date"])
    if verbose and before_dup > len(df):
        print(f"Removed {before_dup - len(df)} duplicate rows")

    # Drop missing values
    before_na = len(df)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    if verbose and before_na > len(df):
        print(f"Removed {before_na - len(df)} rows with missing values")

    # Run data quality audit before removing bad data
    if verbose:
        print("\n--- DATA QUALITY AUDIT ---")
        audit_data_quality(df, verbose=True)

    # Remove rows where low > high (impossible price logic)
    before_logic = len(df)
    df = df[df['low'] <= df['high']]
    if verbose and before_logic > len(df):
        print(f"\nRemoved {before_logic - len(df)} rows where Low > High")

    # Remove non-positive values for all price and volume columns
    for c in ["open", "high", "low", "close", "volume"]:
        before = len(df)
        df = df[df[c] > 0]
        if verbose and before > len(df):
            print(f"Removed {before - len(df)} rows with non-positive {c}")

    # Sort by stock name and date
    df = df.sort_values(["Name", "date"])
    
    # Rename 'Name' to 'ticker' for consistency
    df = df.rename(columns={'Name': 'ticker'})
    
    if verbose:
        print("\n--- FINAL DATA REPORT ---")
        print(f"Total Rows: {len(df)}")
        print(f"Missing Values: {df.isnull().sum().sum()}")
        print(f"Unique Stocks: {df['ticker'].nunique()}")
        print(f"\nVerification:")
        print(f"Remaining Logical Errors: {len(df[df['low'] > df['high']])}")
        print(f"Remaining Zero/Negative Volume: {len(df[df['volume'] <= 0])}")
        print(f"Remaining Zero/Negative Prices: {len(df[df['close'] <= 0])}")
    
    return df

def main():
    os.makedirs("Results/processed", exist_ok=True)
    src = "Dataset/all_stocks_5yr.csv"
    
    print("=" * 50)
    print("DATA CLEANING PROCESS")
    print("=" * 50)
    
    df = pd.read_csv(src)
    print(f"\nLoaded {len(df)} rows from {src}")
    
    df_clean = clean_prices(df, verbose=True)
    
    output_path = "Results/processed/clean_prices.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Saved cleaned data to {output_path}")
    print("\nSample of cleaned data:")
    print(df_clean.head())

if __name__ == "__main__":
    main()