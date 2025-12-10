import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import sys

try:
    import cvxpy as cp
except ImportError:
    print("[WARNING] CVXPY not found. Optimization functions will be stubbed.")
    class MockCVXPY:
        class Variable:
            def __init__(self, n, boolean=False): self.value = np.zeros(n)
        class Problem:
            def __init__(self, objective, constraints): pass
            def solve(self, solver=None, verbose=False): pass
        class Minimize:
            def __init__(self, obj): pass
        def quad_form(self, x, Sigma): return 0
        def sum(self, x): return 0
        def power(self, x, p): return 0
    cp = MockCVXPY()


def load_data():
    """MOCK: Generates simple data for demonstration, with added noise to prevent collinearity errors."""
    N = 10
    base_mu = np.linspace(0.0001, 0.0005, N)
    noise = np.random.normal(0, 0.00005, N) 
    mu = pd.Series(base_mu + noise, index=[f'ASSET_{i}' for i in range(N)])
    
    A = np.random.rand(N, N) * 0.0001
    cov = pd.DataFrame(A @ A.T + np.diag(np.random.rand(N) * 0.00001))
    cov.columns = mu.index
    cov.index = mu.index
    return mu, cov, mu.index.tolist()

def analyze_convexity(Sigma, mu_vec, lam):
    """MOCK: Returns a dictionary indicating convexity status."""
    return {"is_dcp": True, "min_eigenvalue": np.min(np.linalg.eigvalsh(Sigma))}


def make_nonconvex(data, strength):
    """
    Apply a wavy distortion to destroy the convexity of the dataset points.
    data shape: (n, 2) -> [index, return]
    """
    x = data[:, 0]
    y = data[:, 1]

    distortion = strength * np.sin(4 * np.pi * x / len(x))

    y_distorted = y + distortion
    y_distorted = np.maximum(y_distorted, 0)

    return np.column_stack((x, y_distorted))

def plot_dataset_with_hull(data, title_suffix, filename, color):
    """
    Plot the dataset and its convex hull, and save the figure.
    """
    OUTPUT_PLOTS_DIR = 'Results/plots'
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(OUTPUT_PLOTS_DIR, filename)

    plt.figure(figsize=(10,6))
    
    if data.shape[0] >= 3:
        hull = ConvexHull(data)
    else:
        print("[WARNING] Not enough points for Convex Hull.")
        hull = None

    plt.scatter(data[:,0], data[:,1], color=color, label=f"Dataset {title_suffix}", s=80, alpha=0.7)

    if hull:
        hull_points = data[hull.vertices,:]
        hull_points = np.append(hull_points, [hull_points[0]], axis=0) 
        plt.plot(hull_points[:,0], hull_points[:,1], "r-", linewidth=2, alpha=0.5, label='Convex Hull Boundary')

    plt.title(f"Dataset Convexity Check: {title_suffix}", fontsize=15)
    plt.xlabel("Asset Index", fontsize=12)
    plt.ylabel("Return Value (Scaled)", fontsize=12)
    plt.legend()
    plt.tight_layout()

    plt.savefig(plot_path, dpi=300)
    print(f"\n[SUCCESS] Plot saved as: {plot_path}")
    
    plt.close()


def optimize_nonconvex(mu, cov, tickers, lam=10, max_alloc=0.3, k=None):
    """
    Optimize a portfolio with a non-convex objective.
    If exact Mixed-Integer solvers are missing, falls back to a heuristic solution.
    """
    n = len(tickers)
    mu_vec = mu.loc[tickers].values
    Sigma = cov.loc[tickers, tickers].values + 1e-8 * np.eye(n)

    convexity = analyze_convexity(Sigma, mu_vec, lam)

    try:
        x = cp.Variable(n)
        z = cp.Variable(n, boolean=True) if k is not None else None

        cubic_penalty = 0.5 * cp.sum(cp.power(x, 3))
        objective = cp.Minimize(cp.quad_form(x, Sigma) - lam * (mu_vec @ x) + cubic_penalty)

        constraints = [cp.sum(x) == 1, x >= 0, x <= max_alloc]

        if k is not None:
            constraints.append(cp.sum(z) <= k)
            constraints.append(x <= max_alloc * z)

        problem = cp.Problem(objective, constraints)
        
        if k is not None:
             problem.solve(solver=cp.SCIPY) 
        else:
            problem.solve(solver=cp.SCS)

        if x.value is None:
            raise RuntimeError("Solver returned None")
            
        print("Strict solver successful (or mocked).")
        weights = x.value
        status = problem.status

    except (Exception) as e:
        print(f"\n[INFO] Optimization skipped/failed ({e}). Using simple placeholder weights.")
        weights = np.ones(n) / n 
        status = "Mocked/Fallback"

    portfolio_return = float(mu_vec @ weights)
    portfolio_risk = float(np.sqrt(weights @ Sigma @ weights)) if n > 0 else 0
    sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

    return weights, 0.0, {
        "status": status,
        "portfolio_return": portfolio_return,
        "portfolio_risk": portfolio_risk,
        "sharpe_ratio": sharpe_ratio,
        "convexity": convexity,
    }


if __name__ == '__main__':
    np.random.seed(42)
    
    mu, cov, tickers = load_data()
    n_assets = len(tickers)

    asset_index = np.arange(n_assets)
    mean_returns_scaled = mu.loc[tickers].values * 1000 
    initial_dataset = np.column_stack((asset_index, mean_returns_scaled))

    print("\nSTEP 1: Modifying Dataset to Non-Convex and Plotting...")
    
    nonconvex_data = make_nonconvex(
        initial_dataset.copy(), 
        strength=np.max(mean_returns_scaled) / 2
    ) 
    
    plot_dataset_with_hull(
        data=nonconvex_data, 
        title_suffix="Modified (Non-Convex) Dataset",
        filename="nonconvex_dataset_plot.png",
        color="green"
    )
    # 

    print("\nSTEP 2: Running Non-Convex Portfolio Optimization...")
    weights, obj_val, info = optimize_nonconvex(mu, cov, tickers, lam=10, max_alloc=0.3, k=5)

    print("\nOptimization Status:", info["status"])
    print(f"Portfolio Return: {info['portfolio_return']:.4%}")
    print(f"Portfolio Risk:   {info['portfolio_risk']:.4%}")
    print(f"Sharpe Ratio:     {info['sharpe_ratio']:.4f}")
    
    active_count = np.sum(weights > 1e-4)
    print(f"Number of Active Assets: {active_count}")