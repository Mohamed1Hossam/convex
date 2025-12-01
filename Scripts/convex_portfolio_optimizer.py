# Scripts/convex_portfolio_optimizer.py
import os
import json
import numpy as np
import pandas as pd
import cvxpy as cp

def load_data():
    mu = pd.read_json("../Results/processed/mean_returns.json", typ="series")
    cov = pd.read_csv("../Results/processed/cov_matrix.csv", index_col=0)
    tickers = pd.read_csv("../Results/processed/selected_tickers.csv", header=None)[0].tolist()
    return mu, cov, tickers

def optimize_portfolio(mu, cov, tickers, lam=10, max_alloc=0.3):
    mu_vec = mu.loc[tickers].values
    n = len(tickers)
    Sigma = cov.loc[tickers, tickers].values

    # Numerical stability
    Sigma = Sigma + 1e-8 * np.eye(n)

    x = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(x, Sigma) - lam * (mu_vec @ x))
    constraints = [cp.sum(x) == 1, x >= 0, x <= max_alloc]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, verbose=False)

    if x.value is None:
        raise RuntimeError("Optimization failed – check data alignment/PSD.")

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
    print("Convex portfolio optimization complete.")
    print("Output saved to ../Results/optimized_portfolios/convex_solution.json")