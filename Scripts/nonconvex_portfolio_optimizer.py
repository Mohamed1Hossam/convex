import numpy as np
import pandas as pd
import cvxpy as cp
from convex_portfolio_optimizer import analyze_convexity, load_data

# ---------------------- Step 1: Non-convex optimizer ---------------------- #
def optimize_nonconvex(mu, cov, tickers, lam=10, max_alloc=0.3, k=None):
    """
    Optimize a portfolio with a non-convex objective.
    If exact Mixed-Integer solvers are missing, falls back to a heuristic solution.
    """
    n = len(tickers)
    mu_vec = mu.loc[tickers].values
    Sigma = cov.loc[tickers, tickers].values + 1e-8 * np.eye(n)

    # Analyze convexity (for reference)
    convexity = analyze_convexity(Sigma, mu_vec, lam)

    # ---------------------------------------------------------------------
    # ATTEMPT 1: Strict Mixed-Integer Formulation (The "Ideal" Model)
    # ---------------------------------------------------------------------
    try:
        x = cp.Variable(n)
        z = cp.Variable(n, boolean=True) if k is not None else None

        # Objective: Risk - Return + Cubic Term (Non-Linear)
        # Note: cp.power(x,3) is used for solver compatibility
        cubic_penalty = 0.5 * cp.sum(cp.power(x, 3))
        objective = cp.Minimize(cp.quad_form(x, Sigma) - lam * (mu_vec @ x) + cubic_penalty)

        # Constraints
        constraints = [cp.sum(x) == 1, x >= 0, x <= max_alloc]

        if k is not None:
            # Cardinality Constraints
            constraints.append(cp.sum(z) <= k)
            constraints.append(x <= max_alloc * z)

        problem = cp.Problem(objective, constraints)
        
        # Try to solve. 
        # Note: This will raise SolverError if no MIQP solver (like Gurobi/CPLEX) is installed.
        if k is not None:
             # Try using SCIPY or GLPK_MI if available, though they often struggle with MIQP
            problem.solve(solver=cp.SCIPY) 
        else:
            problem.solve(solver=cp.SCS)

        if x.value is None:
            raise RuntimeError("Solver returned None")
            
        print("Strict solver successful.")
        weights = x.value
        status = problem.status

    # ---------------------------------------------------------------------
    # ATTEMPT 2: Fallback Heuristic (If Solver fails)
    # ---------------------------------------------------------------------
    except (cp.error.SolverError, RuntimeError) as e:
        print(f"\n[INFO] Strict MIQP Solver failed ({e}).")
        print("[INFO] Switching to Heuristic Cardinality Solution (Select Top K -> Optimize)...")

        # 1. Heuristic Selection: Pick top K assets by simple Sharpe estimate (Return/StdDev)
        # (This mimics the integer decision variable z)
        variances = np.diag(Sigma)
        simple_sharpe = mu_vec / np.sqrt(variances)
        
        if k is not None:
            top_k_indices = np.argsort(simple_sharpe)[-k:]
            active_mask = np.zeros(n)
            active_mask[top_k_indices] = 1.0
        else:
            active_mask = np.ones(n)

        # 2. Optimize weights for ONLY the selected assets
        # This reduces the problem to a standard Convex problem (QP) which SCS/OSQP can solve easily.
        x_fallback = cp.Variable(n)
        
        # Same objective
        cubic_penalty_fb = 0.5 * cp.sum(cp.power(x_fallback, 3))
        obj_fb = cp.Minimize(cp.quad_form(x_fallback, Sigma) - lam * (mu_vec @ x_fallback) + cubic_penalty_fb)
        
        # Constraints: Force non-selected assets to 0
        cons_fb = [
            cp.sum(x_fallback) == 1, 
            x_fallback >= 0, 
            x_fallback <= max_alloc
        ]
        
        # Enforce cardinality manually by setting weights to 0 for non-top-k
        # (Inverse of the mask)
        for i in range(n):
            if active_mask[i] == 0:
                cons_fb.append(x_fallback[i] == 0)

        problem = cp.Problem(obj_fb, cons_fb)
        problem.solve(solver=cp.SCS, verbose=False) # SCS is standard and robust for this
        
        if x_fallback.value is None:
             raise RuntimeError("Heuristic Optimization also failed.")
             
        weights = x_fallback.value
        status = "Heuristic Optimal"

    # ---------------------------------------------------------------------
    # Compute Final Metrics
    # ---------------------------------------------------------------------
    portfolio_return = float(mu_vec @ weights)
    portfolio_risk = float(np.sqrt(weights @ Sigma @ weights))
    sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

    return weights, 0.0, {
        "status": status,
        "portfolio_return": portfolio_return,
        "portfolio_risk": portfolio_risk,
        "sharpe_ratio": sharpe_ratio,
        "convexity": convexity,
    }

# ---------------------- Step 2: Main execution ---------------------- #
if __name__ == '__main__':
    mu, cov, tickers = load_data()

    print(f"Running Non-Convex Model (Cardinality k=5)...")
    weights, obj_val, info = optimize_nonconvex(mu, cov, tickers, lam=10, max_alloc=0.3, k=5)

    print("\nOptimization Status:", info["status"])
    print(f"Portfolio Return: {info['portfolio_return']:.4%}")
    print(f"Portfolio Risk:   {info['portfolio_risk']:.4%}")
    print(f"Sharpe Ratio:     {info['sharpe_ratio']:.4f}")
    
    # Check Active Assets
    active_count = np.sum(weights > 1e-4)
    print(f"Number of Active Assets: {active_count}")