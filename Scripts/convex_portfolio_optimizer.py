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
    # The objective function is convex if its Hessian (which is 2*Sigma) is Positive Semi-Definite (PSD).
    # This is true if all eigenvalues of the Hessian (and thus Sigma) are non-negative.
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
    
    # Check if the required columns exist
    if 'return' not in data_df.columns or 'risk' not in data_df.columns:
        print("[SKIP] Cannot plot convex hull: Required 'return' and 'risk' columns not found.")
        print("Note: This function expects a DataFrame of individual stock metrics (return/risk).")
        return

    # Convert the required columns to a NumPy array for ConvexHull
    points = data_df[['risk', 'return']].values

    # Compute the convex hull
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
    """
    Plots a 2D slice of the objective function to visually inspect its convexity.
    """
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
        # Objective: x^T * Sigma * x - lam * mu^T * x
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
    # Define variables
    x = cp.Variable(n)
    
    # Objective function (DCP compliant)
    # This minimizes the risk-adjusted return (Risk - lambda*Return)
    objective = cp.Minimize(cp.quad_form(x, Sigma) - lam * (mu_vec @ x))
    
    constraints = [
        cp.sum(x) == 1, 
        x >= 0,   
        x <= max_alloc
    ]

    # Form the CVXPY problem
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

# ---------------------- REPORT WRITING ----------------------
def write_report(problem_info, objective_value, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("CONVEX PORTFOLIO OPTIMIZATION - DETAILED REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. PROBLEM FORMULATION (CVXPY)\n")
        f.write("-" * 70 + "\n")
        f.write("Objective: minimize f(x) = x^T Σx - λμ^T x\n")
        f.write("Constraints:\n")
        f.write("  - Sum(x_i) = 1 (Full Investment)\n")
        f.write("  - x_i >= 0 (No Short Selling)\n")
        f.write("  - x_i <= Max Allocation (Concentration Limit)\n\n")

        cv = problem_info["convexity"]
        f.write("2. CONVEXITY ANALYSIS (Objective Function)\n")
        f.write("-" * 70 + "\n")
        
        # New: Second derivative check (Hessian)
        f.write("2.1 Second Derivative (Hessian) Check:\n")
        f.write(f"Hessian = 2 * Covariance Matrix (Σ)\n")
        f.write(f"Minimum eigenvalue of Σ: {cv['min_eigenvalue']:.6e}\n")
        f.write(f"Is Σ Positive Semi-Definite (PSD)?: {cv['matrix_psd']}\n\n")
        
        # New: CVXPY DCP check
        f.write("2.2 CVXPY (DCP) Rules Check:\n")
        f.write(f"Objective curvature: {cv['objective_curvature']}\n")
        f.write(f"DCP compliant: {cv['is_dcp']}\n\n")
        
        # Retain Mathematical Justification
        f.write("2.3 MATHEMATICAL JUSTIFICATION\n")
        f.write("-" * 70 + "\n")
        f.write(
            "The objective function is composed of two parts:\n\n"
            "1) Quadratic risk term: x^T Σ x\n"
            "   • A quadratic form x^T Σ x is convex if and only if Σ is positive semi-definite (PSD).\n"
            f"   • From the eigenvalue analysis, the minimum eigenvalue of Σ is {'non-negative' if cv['matrix_psd'] else 'negative'}, confirming PSD.\n"
            "   • Therefore, the risk term is convex.\n\n"
            "2) Linear return term: -λ μ^T x\n"
            "   • Linear (affine) functions are both convex and concave.\n"
            "   • Minimizing a convex function (Risk) plus an affine function (Negative Return) is a convex problem.\n\n"
            "The constraints are all linear, which defines a convex feasible set.\n"
            "Thus, the entire optimization problem is convex.\n\n"
        )

        f.write("3. OPTIMIZATION RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Status: {problem_info['status']}\n")
        f.write(f"Objective value: {objective_value:.6f}\n")
        f.write(f"Portfolio return: {problem_info['portfolio_return']:.4%}\n")
        f.write(f"Portfolio risk: {problem_info['portfolio_risk']:.4%}\n")
        f.write(f"Sharpe ratio: {problem_info['sharpe_ratio']:.4f}\n")

# ---------------------- MAIN EXECUTION ----------------------
if __name__ == "__main__":
    mu, cov, tickers = load_data()
    weights, obj_val, info = optimize_portfolio(mu, cov, tickers)

    os.makedirs("Results/optimized_portfolios", exist_ok=True)
    report_path = "Results/optimized_portfolios/convexity_report.txt"

    write_report(info, obj_val, report_path)
    print(f"\n[SUCCESS] Convex Optimization complete. Report saved to {report_path}")