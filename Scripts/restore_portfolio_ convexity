"""
restore_convex_portfolio_optimizer.py
Restores convexity after using the non-convex portfolio model.
This solves the original convex Markowitz model:

    Minimize: x^T Σ x - λ μ^T x
    Subject to: sum(x)=1, x>=0, x<=max_alloc

This file is designed to work directly after running:
    nonconvex_portfolio_optimizer.py
"""

import numpy as np
import cvxpy as cp
import pandas as pd
from convex_portfolio_optimizer import analyze_convexity, load_data


def restore_convex(mu, cov, tickers, lam=10, max_alloc=0.3):
    """
    Restore convexity by removing non-convex terms
    and solving the standard convex Markowitz portfolio model.

    Parameters:
        mu (Series): expected returns
        cov (DataFrame): covariance matrix
        tickers (list): list of tickers
        lam (float): risk-return tradeoff parameter
        max_alloc (float): maximum allocation per asset

    Returns:
        weights (np.array)
        objective_value (float)
        info (dict)
    """
    n = len(tickers)

    # Extract arrays
    mu_vec = mu.loc[tickers].values
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


# ---------------------- Testing Script ---------------------- #
if __name__ == "__main__":
    mu, cov, tickers = load_data()

    weights, obj_val, info = restore_convex(mu, cov, tickers, lam=10, max_alloc=0.3)

    print("\n--- RESTORED CONVEX PORTFOLIO ---")
    print("Status:", info["status"])
    print("Return:", info["portfolio_return"])
    print("Risk:", info["portfolio_risk"])
    print("Sharpe Ratio:", info["sharpe_ratio"])
    print("Weights:", weights)
    print("\nConvexity Check:", info["convexity_after_restore"])
