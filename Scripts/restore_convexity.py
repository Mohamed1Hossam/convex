"""restore_convexity.py
Methods to restore or approximate convexity for nonconvex solutions.
"""
import numpy as np
from convex_portfolio_optimizer import optimize_portfolio

def project_to_simplex(weights):
    # simple projection onto probability simplex
    w = np.maximum(weights, 0)
    s = w.sum()
    if s == 0:
        return np.ones_like(w)/len(w)
    return w / s

def restore(weights, expected_returns, cov_matrix):
    # Project onto simplex then run convex optimizer with target_return
    w_proj = project_to_simplex(weights)
    target_ret = w_proj @ expected_returns
    w_convex, obj = optimize_convex(expected_returns, cov_matrix, target_return=target_ret)
    return w_convex, obj

if __name__ == '__main__':
    print('Provides restore(weights, expected_returns, cov_matrix)')
