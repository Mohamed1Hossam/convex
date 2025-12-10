import numpy as np
import cvxpy as cp
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from convex_portfolio_optimizer import analyze_convexity, load_data


# ---------------------- CONVEX HULL AND PLOTTING FUNCTION ----------------------

def plot_convex_hull(mu, Sigma, tickers, plot_path="Results/plots/dataset_convex_hull_restored.png"):
    # Risk is standard deviation (sqrt of diagonal of Covariance matrix)
    risk = np.sqrt(np.diag(Sigma))
    data_df = pd.DataFrame({'return': mu, 'risk': risk}, index=tickers)
    points = data_df[['risk', 'return']].values

    if points.shape[0] < 3:
        print("[SKIP] Cannot plot convex hull: Need at least 3 assets.")
        return

    # Compute the convex hull
    hull = ConvexHull(points)

    plt.figure(figsize=(10, 8))
    
    # 1. Plot all data points (individual stocks)
    plt.plot(points[:, 0], points[:, 1], 'o', markersize=4, label='Individual Stock Risk-Return')
    
    # 2. Plot the convex hull (boundary)
    # Simplex indices define the edges of the hull
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'r-', linewidth=2, alpha=0.7)
    
    plt.title('Restored Convexity Check: Asset Universe (Risk vs. Return)', fontsize=15)
    plt.xlabel('Risk (Standard Deviation)', fontsize=12)
    plt.ylabel('Return (Mean Daily Return)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"\n[SUCCESS] Convex Hull plot saved to {plot_path}")


# ---------------------- RESTORE CONVEX OPTIMIZATION ----------------------

def restore_convex(mu, cov, tickers, lam=10, max_alloc=0.3):
    n = len(tickers)

    # Extract arrays
    mu_vec = mu.loc[tickers].values
    # Add small identity matrix for positive semi-definiteness
    Sigma = cov.loc[tickers, tickers].values + 1e-8 * np.eye(n)

    # Analyze convexity (for reporting)
    convexity = analyze_convexity(Sigma, mu_vec, lam)

    # Variables
    x = cp.Variable(n)

    # Pure convex objective (restored): variance - λ * return
    objective = cp.Minimize(cp.quad_form(x, Sigma) - lam * (mu_vec @ x))

    # Convex constraints
    constraints = [
        cp.sum(x) == 1,
        x >= 0,
        x <= max_alloc
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, verbose=False)

    if x.value is None:
        raise RuntimeError("Convex restored optimization failed.")

    weights = x.value
    portfolio_return = float(mu_vec @ weights)
    portfolio_risk = float(np.sqrt(weights @ Sigma @ weights))
    sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

    info = {
        "status": problem.status,
        "portfolio_return": portfolio_return,
        "portfolio_risk": portfolio_risk,
        "sharpe_ratio": sharpe_ratio,
        "restoration_method": "Removed cubic + cardinality to enforce convexity",
        "convexity_after_restore": convexity
    }

    return weights, problem.value, info


if __name__ == "__main__":
    mu, cov, tickers = load_data()
    n = len(tickers)
    
    # Calculate Sigma for plotting and analysis
    Sigma = cov.loc[tickers, tickers].values + 1e-8 * np.eye(n)

    print("\nPlotting Restored Convex Dataset...")
    plot_convex_hull(mu.loc[tickers].values, Sigma, tickers)

    weights, obj_val, info = restore_convex(mu, cov, tickers, lam=10, max_alloc=0.3)

    print("\n--- RESTORED CONVEX PORTFOLIO ---")
    print("Status:", info["status"])
    print("Return:", info["portfolio_return"])
    print("Risk:", info["portfolio_risk"])
    print("Sharpe Ratio:", info["sharpe_ratio"])
    print("Weights:", weights)
    print("\nConvexity Check:", info["convexity_after_restore"])