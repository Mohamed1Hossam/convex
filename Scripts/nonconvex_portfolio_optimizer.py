"""nonconvex_portfolio_optimizer.py
Example non-convex optimizer (e.g., cardinality constraints) using SciPy.
"""
import numpy as np
from scipy.optimize import minimize

def optimize_nonconvex(expected_returns, cov_matrix, k=None):
    n = len(expected_returns)
    # simple continuous relaxation placeholder
    def obj(w):
        return w.T @ cov_matrix @ w
    cons = ({'type':'eq', 'fun': lambda w: np.sum(w)-1},)
    bounds = [(0,1) for _ in range(n)]
    x0 = np.ones(n)/n
    res = minimize(obj, x0, bounds=bounds, constraints=cons)
    return res.x, res.fun

if __name__ == '__main__':
    print('Provides optimize_nonconvex(expected_returns, cov_matrix, k=None)')
