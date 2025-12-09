import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import seaborn as sns

OUTPUT_PLOTS_DIR = 'Results/plots'
RESULTS_JSON_DIR = 'Results/optimized_portfolios'

def plot_three_cases_comparison(results_dir=RESULTS_JSON_DIR, out_path=os.path.join(OUTPUT_PLOTS_DIR, "comparison_risk_return.png")):
    print("STARTING RISK-RETURN COMPARISON VISUALIZATION")

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
            print(f"Warning: {filename} not found in {results_dir}. Skipping point.")

    if not data:
        print("[SKIP] No solution data found to plot the comparison. Skipping plot generation.")
        return

    df_plot = pd.DataFrame(data)
    
    # 2. Plotting (Seaborn Scatter Plot)
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
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
    
    print("VISUALIZATION GENERATION COMPLETE")


if __name__ == '__main__':
    plot_three_cases_comparison()