"""
nonconvex_portfolio_optimizer.py
Non-convex portfolio optimization by modifying the convex model.
Includes:
- Cubic term in the objective to introduce non-convexity
- Optional cardinality constraint (number of assets ≤ k)
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from convex_portfolio_optimizer import analyze_convexity, load_data  # correct import

# ---------------------- Step 1: Non-convex optimizer ---------------------- #
def optimize_nonconvex(mu, cov, tickers, lam=10, max_alloc=0.3, k=None):
    """
    Optimize a portfolio with a non-convex objective.
    - mu: expected returns (Series)
    - cov: covariance matrix (DataFrame)
    - tickers: list of asset tickers
    - lam: risk-return tradeoff parameter
    - max_alloc: maximum allocation per asset
    - k: optional cardinality constraint (number of assets)
    """
    n = len(tickers)
    x = cp.Variable(n)
    mu_vec = mu.loc[tickers].values
    Sigma = cov.loc[tickers, tickers].values + 1e-8 * np.eye(n)

    # Analyze convexity (for reference)
    convexity = analyze_convexity(Sigma, mu_vec, lam)

    # Non-convex objective: quadratic risk - linear return + cubic term
    objective = cp.Minimize(cp.quad_form(x, Sigma) - lam * (mu_vec @ x) + 0.5 * cp.sum(x**3))

    # Constraints: sum to 1, non-negative, max allocation
    constraints = [cp.sum(x) == 1, x >= 0, x <= max_alloc]

    # Optional cardinality constraint (number of non-zero assets ≤ k)
    if k is not None:
        constraints.append(cp.sum(cp.sign(x)) <= k)

    # Solve using SCS solver (more stable for non-convex problems)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=True)

    if x.value is None:
        raise RuntimeError("Optimization failed.")

    # Compute portfolio metrics
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

# ---------------------- Step 2: Main execution ---------------------- #
if __name__ == '__main__':
    mu, cov, tickers = load_data()

    # Solve non-convex portfolio without cardinality
    weights, obj_val, info = optimize_nonconvex(mu, cov, tickers, lam=10, max_alloc=0.3)

    print("Optimization Status:", info["status"])
    print("Portfolio Return:", info["portfolio_return"])
    print("Portfolio Risk:", info["portfolio_risk"])
    print("Sharpe Ratio:", info["sharpe_ratio"])
    print("Portfolio Weights:", weights)
    print("Convexity Analysis:", info["convexity"])