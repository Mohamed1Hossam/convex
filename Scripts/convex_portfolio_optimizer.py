import pandas as pd
import numpy as np
import cvxpy as cp
import json
import os

def load_data():
    mu = pd.read_json("../Results/processed/mean_returns.json", typ="series")
    cov = pd.read_csv("../Results/processed/cov_matrix.csv", index_col=0)
    tickers = pd.read_csv("../Results/processed/selected_tickers.csv", header=None)[0].tolist()

    return mu, cov, tickers

def optimize_portfolio(mu, cov, tickers, lam=10, max_alloc=0.3):
    n = len(tickers)

    # Variable
    x = cp.Variable(n)

    # Convert to arrays
    Sigma = cov.values
    mu_vec = mu.values

    # Objective: xᵀ Σ x − λ μᵀ x
    objective = cp.Minimize(cp.quad_form(x, Sigma) - lam * mu_vec @ x)

    # Constraints
    constraints = [
        cp.sum(x) == 1,
        x >= 0,
        x <= max_alloc
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return x.value, problem.value

def save_solution(tickers, weights, objective_value):
    os.makedirs("../Results/optimized_portfolios", exist_ok=True)
    result = {
        "tickers": tickers,
        "weights": weights.tolist(),
        "objective_value": float(objective_value)
    }
    with open("../Results/optimized_portfolios/convex_solution.json", "w") as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    mu, cov, tickers = load_data()

    weights, obj_val = optimize_portfolio(mu, cov, tickers)

    save_solution(tickers, weights, obj_val)

    print("✔ Convex portfolio optimization complete.")
    print("✔ Output saved to ../Results/optimized_portfolios/convex_solution.json")
