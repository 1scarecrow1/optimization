import numpy as np
from presto.line_search.line_search import backtracking
from presto.linalg import l2_norm

def augmented_lagrangian(func, x, constraints, grad, jac, bounds,
                         lambdas, mu0=10.0, eta_tol=1e-6,
                         omega_tol=1e-6, max_iter=100,
                         subproblem_max_iter=200):
    x_cur = x.copy()
    lambda_cur = lambdas.copy()
    lower = np.array([-np.inf if pair[0] is None else pair[0] for pair in bounds])
    upper = np.array([np.inf if pair[1] is None else pair[1] for pair in bounds])
    mu_cur = mu0
    omega_cur = 1 / mu_cur
    eta_cur = 1 / mu_cur**0.1

    for _ in range(max_iter):
        def objective(z):
            cz = constraints(z)
            return func(z) - lambda_cur @ cz + 0.5 * mu_cur * (cz @ cz)

        def objective_grad(z):
            cz = constraints(z)
            Az = jac(z)
            return grad(z) - Az.T @ lambda_cur + mu_cur * Az.T @ cz

        for _ in range(subproblem_max_iter):
            grad_aug = objective_grad(x_cur)
            p_cur = np.clip(x_cur - grad_aug, lower, upper) - x_cur
            
            if l2_norm(p_cur) <= omega_cur:
                break

            _, x_cur, _ = backtracking(objective, x_cur, objective(x_cur),
                                       grad_aug, p_cur, a0=1.0, c1=1e-4,
                                       rho=0.5, max_iter=50)

        c_cur = constraints(x_cur)
        grad_aug = objective_grad(x_cur)
        projected_grad = x_cur - np.clip(x_cur - grad_aug, lower, upper)

        if l2_norm(c_cur) <= eta_cur:
            if l2_norm(c_cur) <= eta_tol and l2_norm(projected_grad) <= omega_tol:
                return x_cur
            lambda_cur = lambda_cur - mu_cur * c_cur
            eta_cur = eta_cur / mu_cur**0.9
            omega_cur = omega_cur / mu_cur
        else:
            mu_cur = 100 * mu_cur
            eta_cur = 1 / mu_cur**0.1
            omega_cur = 1 / mu_cur

    return x_cur
