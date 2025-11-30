"""convex_portfolio_optimizer.py
Mean-variance optimizer using cvxpy.
"""
import numpy as np
import pandas as pd
import cvxpy as cp

def optimize_convex(expected_returns, cov_matrix, target_return=None, reg=1e-6):
    n = len(expected_returns)
    w = cp.Variable(n)
    mu = expected_returns
    Sigma = cov_matrix

    obj = cp.quad_form(w, Sigma)  # minimize variance
    constraints = [cp.sum(w) == 1, w >= 0]
    if target_return is not None:
        constraints.append(w @ mu >= target_return)

    prob = cp.Problem(cp.Minimize(obj + reg*cp.norm(w,2)), constraints)
    prob.solve(solver=cp.SCS)
    return np.array(w.value).flatten(), prob.value

if __name__ == '__main__':
    print('This module provides optimize_convex(expected_returns, cov_matrix, target_return=None)')
