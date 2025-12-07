import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import seaborn as sns

## 📊 Function 1: Visualize Financial Trends

def plot_price_trends(price_df, out_path):
    """
    Plots the normalized price movement for selected assets.
    """
    plt.figure(figsize=(12,6))
    for col in price_df.columns:
        # Normalize prices by the starting price (iloc[0])
        norm = price_df[col] / price_df[col].iloc[0]
        plt.plot(price_df.index, norm, label=col, alpha=0.6)
    
    plt.title('Normalized Price Trends of Selected Assets', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Normalized Price (Start=1.0)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"[SUCCESS] Price trends plot saved to {out_path}")
    plt.close()

# ---------------------------------------------------------------------------

## 📈 Function 2: Risk-Return Tradeoff Comparison (Three Cases)

def plot_three_cases_comparison(results_dir="Results/optimized_portfolios", out_path="Results/plots/comparison_risk_return.png"):
    """
    Loads JSON results from the 3 models (Convex, Non-Convex, Restored) and plots 
    their Risk-Return tradeoff on a single scatter plot.
    
    This fulfills the requirement: "Use graphs to display the relationship between 
    risk and return in the three cases."
    """
    # File names expected
    files = {
        "Convex (Original)": "convex_solution.json",
        "Non-Convex (Cardinality)": "nonconvex_solution.json",
        "Restored Convex": "restored_solution.json"
    }
    
    data = []
    
    # 1. Load Data from JSON files
    for label, filename in files.items():
        path = os.path.join(results_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    res = json.load(f)
                data.append({
                    "Model": label,
                    "Return": res['portfolio_return'],
                    "Risk": res['portfolio_risk'],
                    "Sharpe": res['sharpe_ratio']
                })
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Warning: {filename} not found in {results_dir}")

    if not data:
        print("No data found to plot. Skipping plot generation.")
        return

    df_plot = pd.DataFrame(data)
    
    # 2. Plotting (Seaborn Scatter Plot)
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Create Scatter plot 
    sns.scatterplot(
        data=df_plot, 
        x="Risk", 
        y="Return", 
        hue="Model", 
        style="Model", 
        s=300, # Marker size
        palette="deep"
    )
    
    # 3. Annotate points with Model Name and Sharpe Ratio (SR)
    for i in range(df_plot.shape[0]):
        plt.text(
            df_plot.Risk.iloc[i] + 0.0002, 
            df_plot.Return.iloc[i], 
            f"{df_plot.Model.iloc[i]}\nSR: {df_plot.Sharpe.iloc[i]:.2f}", 
            horizontalalignment='left', 
            size='medium', 
            color='black', 
            weight='semibold'
        )

    plt.title("Portfolio Optimization Comparison: Convex vs Non-Convex vs Restored", fontsize=15)
    plt.xlabel("Portfolio Risk (Standard Deviation)", fontsize=12)
    plt.ylabel("Expected Portfolio Return", fontsize=12)
    plt.legend(loc='lower right')
    
    # 4. Save Plot
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"[SUCCESS] Comparison plot saved to {out_path}")
    plt.close()

if __name__ == '__main__':
    # Test run
    plot_three_cases_comparison()