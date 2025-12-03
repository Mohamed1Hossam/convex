import os
import json
import numpy as np
import pandas as pd
import cvxpy as cp

def load_data():
    mu = pd.read_json("Results/processed/mean_returns.json", typ="series")
    cov = pd.read_csv("Results/processed/cov_matrix.csv", index_col=0)
    tickers = pd.read_csv("Results/processed/tickers.txt", header=None)[0].tolist()
    return mu, cov, tickers


def analyze_convexity(Sigma, mu_vec, lam):
    eigenvalues = np.linalg.eigvalsh(Sigma)
    min_eigenvalue = float(np.min(eigenvalues))
    is_psd = min_eigenvalue >= -1e-10

    x_test = cp.Variable(len(mu_vec))
    test_obj = cp.quad_form(x_test, Sigma) - lam * (mu_vec @ x_test)

    return {
        "matrix_dim": Sigma.shape,
        "min_eigenvalue": min_eigenvalue,
        "matrix_psd": is_psd,
        "objective_curvature": test_obj.curvature,
        "is_dcp": test_obj.is_dcp(),
    }


def optimize_portfolio(mu, cov, tickers, lam=10, max_alloc=0.3):
    mu_vec = mu.loc[tickers].values
    n = len(tickers)
    Sigma = cov.loc[tickers, tickers].values + 1e-8 * np.eye(n)

    convexity = analyze_convexity(Sigma, mu_vec, lam)

    x = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(x, Sigma) - lam * (mu_vec @ x))
    constraints = [cp.sum(x) == 1, x >= 0, x <= max_alloc]

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


def write_report(problem_info, objective_value, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("CONVEX PORTFOLIO OPTIMIZATION - DETAILED REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. PROBLEM FORMULATION\n")
        f.write("-" * 70 + "\n")
        f.write("Objective: minimize f(x) = x^T Σx - λμ^T x\n\n")

        cv = problem_info["convexity"]
        f.write("2. CONVEXITY ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Matrix dimension: {cv['matrix_dim']}\n")
        f.write(f"Minimum eigenvalue: {cv['min_eigenvalue']:.6e}\n")
        f.write(f"Matrix PSD: {cv['matrix_psd']}\n")
        f.write(f"Objective curvature: {cv['objective_curvature']}\n")
        f.write(f"DCP compliant: {cv['is_dcp']}\n\n")

        # ----------------------------- #
        # MATHEMATICAL EXPLANATION OF CONVEXITY
        # ----------------------------- #
        f.write("2.1 MATHEMATICAL JUSTIFICATION OF CONVEXITY\n")
        f.write("-" * 70 + "\n")
        f.write(
            "The objective function is composed of two parts:\n\n"
            "1) Quadratic risk term: x^T Σ x\n"
            "   • A quadratic form x^T Σ x is convex if and only if Σ is positive semidefinite (PSD).\n"
            "   • From the eigenvalue analysis, the minimum eigenvalue of Σ is >= 0, confirming PSD.\n"
            "   • Therefore, the risk term is convex.\n\n"
            "2) Linear return term: -λ μ^T x\n"
            "   • Linear (affine) functions are both convex and concave.\n"
            "   • Subtracting a linear term does not change convexity.\n\n"
            "Since a convex function plus an affine function is still convex, the total objective:\n"
            "        f(x) = x^T Σ x - λ μ^T x\n"
            "is convex.\n\n"
            "The constraints (sum(x)=1, x>=0, x<=max_alloc) are linear, and linear constraints preserve convexity.\n"
            "Therefore, the entire optimization problem is convex.\n\n"
        )

        f.write("3. OPTIMIZATION RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Status: {problem_info['status']}\n")
        f.write(f"Objective value: {objective_value:.6f}\n")
        f.write(f"Portfolio return: {problem_info['portfolio_return']:.4%}\n")
        f.write(f"Portfolio risk: {problem_info['portfolio_risk']:.4%}\n")
        f.write(f"Sharpe ratio: {problem_info['sharpe_ratio']:.4f}\n")


if __name__ == "__main__":
    mu, cov, tickers = load_data()
    weights, obj_val, info = optimize_portfolio(mu, cov, tickers)

    os.makedirs("Results/optimized_portfolios", exist_ok=True)
    report_path = "Results/optimized_portfolios/convexity_report.txt"

    write_report(info, obj_val, report_path)
