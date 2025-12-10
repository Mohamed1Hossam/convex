import os
import json
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def load_data():
    mu = pd.read_json("Results/processed/mean_returns.json", typ="series")
    cov = pd.read_csv("Results/processed/cov_matrix.csv", index_col=0)
    tickers = pd.read_csv("Results/processed/tickers.txt", header=None)[0].tolist()
    return mu, cov, tickers

def analyze_convexity(Sigma, mu_vec, lam):

    # 1. Second Derivative Check (Hessian = 2*Sigma)
    eigenvalues = np.linalg.eigvalsh(Sigma)
    min_eigenvalue = float(np.min(eigenvalues))
    is_psd = min_eigenvalue >= -1e-10

    # 2. CVXPY DCP Check
    x_test = cp.Variable(len(mu_vec))
    test_obj = cp.quad_form(x_test, Sigma) - lam * (mu_vec @ x_test)

    return {
        "matrix_dim": Sigma.shape,
        "min_eigenvalue": min_eigenvalue,
        "matrix_psd": is_psd,
        "objective_curvature": test_obj.curvature,
        "is_dcp": test_obj.is_dcp(),
    }

# ---------------------- CONVEX HULL AND PLOTTING FUNCTIONS ----------------------

def plot_convex_hull(data_df, plot_path="Results/plots/dataset_convex_hull.png"):
    if data_df.shape[1] < 2:
        print("[SKIP] Cannot plot convex hull: Data has less than 2 dimensions.")
        return
    
    if 'return' not in data_df.columns or 'risk' not in data_df.columns:
        print("[SKIP] Cannot plot convex hull: Required 'return' and 'risk' columns not found.")
        print("Note: This function expects a DataFrame of individual stock metrics (return/risk).")
        return

    points = data_df[['risk', 'return']].values

    hull = ConvexHull(points)

    plt.figure(figsize=(10, 8))
    
    # 1. Plot all data points (individual stocks)
    plt.plot(points[:, 0], points[:, 1], 'o', markersize=4, label='Individual Stock Risk-Return')
    
    # 2. Plot the convex hull (boundary)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'r-')
    
    plt.title('Dataset Convexity Check (Risk vs. Return)', fontsize=15)
    plt.xlabel('Risk (Standard Deviation)', fontsize=12)
    plt.ylabel('Return (Mean Daily Return)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"[SUCCESS] Convex Hull plot saved to {plot_path}")


def plot_objective_function(Sigma, mu_vec, lam, plot_path="Results/plots/objective_function_convexity.png"):
    n = len(mu_vec)
    if n < 2:
        print("[SKIP] Cannot plot objective function: Need at least 2 assets.")
        return

    # Use only the first two assets
    Sigma_2 = Sigma[:2, :2]
    mu_vec_2 = mu_vec[:2]

    # Define the range for the first weight x1 (from 0 to 1, since x2 = 1 - x1)
    x1_range = np.linspace(0, 1, 100)
    
    # Calculate the objective value for each x1
    obj_values = []
    for x1 in x1_range:
        x2 = 1.0 - x1
        x = np.array([x1, x2])
        risk_term = x.T @ Sigma_2 @ x
        return_term = lam * (mu_vec_2 @ x)
        obj = risk_term - return_term
        obj_values.append(obj)

    plt.figure(figsize=(10, 6))
    plt.plot(x1_range, obj_values, linewidth=3)
    
    plt.title('Objective Function Convexity Check (2D Slice)', fontsize=15)
    plt.xlabel(f'Weight of Asset 1 (x1)', fontsize=12)
    plt.ylabel(r'Objective Value ($\mathbf{x}^T \mathbf{\Sigma} \mathbf{x} - \lambda \mathbf{\mu}^T \mathbf{x}$)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"[SUCCESS] Objective Function plot saved to {plot_path}")

# ---------------------- OPTIMIZATION ----------------------
def optimize_portfolio(mu, cov, tickers, lam=10, max_alloc=0.3):
    mu_vec = mu.loc[tickers].values
    n = len(tickers)
    Sigma = cov.loc[tickers, tickers].values + 1e-8 * np.eye(n)

    # 1. Run Convexity Analysis
    convexity = analyze_convexity(Sigma, mu_vec, lam)

    # 2. CVXPY Optimization (Problem + Objective + Constraints)
    x = cp.Variable(n)
    
    objective = cp.Minimize(cp.quad_form(x, Sigma) - lam * (mu_vec @ x))
    
    constraints = [
        cp.sum(x) == 1, 
        x >= 0,   
        x <= max_alloc
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, verbose=False)

    if x.value is None:
        raise RuntimeError("Optimization failed.")

    weights = x.value
    portfolio_return = float(mu_vec @ weights)
    portfolio_risk = float(np.sqrt(weights @ Sigma @ weights))
    sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

    return weights, problem.value, {
        "status": problem.status,
        "portfolio_return": portfolio_return,
        "portfolio_risk": portfolio_risk,
        "sharpe_ratio": sharpe_ratio,
        "convexity": convexity,
    }

if __name__ == "__main__":
    mu, cov, tickers = load_data()
    weights, obj_val, info = optimize_portfolio(mu, cov, tickers)
