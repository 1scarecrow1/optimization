from functools import partial
import warnings
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import presto.conjugate_gradient.optimize as cg
from presto.gradients import gradient
from presto.minimize_res import MinimizeResult
from presto.linalg import backward_substitution, l2_norm
from presto.utils import resolve_func

def linear_least_squares(func, x0, jac=None,
                 solver='QR', solver_args=None, 
                 conv_tol=1e-5, max_iter=200, **func_args):

    func_args = func_args or {}
    solver_args = solver_args or {}
    solve_lstsq = partial(resolve_func(solver, LINEAR_LSTSQ_METHODS, "least squares", qr_solve), **solver_args)
    lstsq_method = solve_lstsq.func

    x0 = np.asarray(x0, dtype=float)
    r = partial(func, **func_args) 
    J = gradient(r, x0, jac)
    b = J @ x0 - r(x0)
    converged_flag = False
    if lstsq_method in NON_ITERATIVE_SOLVERS: 
        x = solve_lstsq(J, b)
        rx = r(x)
        if converged(J.T @ rx, conv_tol=conv_tol):
            converged_flag = True
        f_min = 0.5*(rx @ rx)
        iterations = None

    else:
        res = solve_lstsq(J, b, x0, disp_full=True, conv_tol=conv_tol, max_iter=max_iter)
        x, f_min, iterations, converged_flag = res.x, res.f_min, res.iterations, res.converged

    return MinimizeResult(
        x=x,
        f_min=f_min,
        iterations=iterations,
        func=None,
        solver='least squares',
        method=solver,
        converged=converged_flag
    )


def cholesky_solve(A, b):
    coef = A.T @ A
    y = A.T @ b
    try:
        cfac = cho_factor(coef)
        return cho_solve(cfac, y)
    except np.linalg.LinAlgError:
        raise ValueError("Cholesky factorization breakdown - ill conditional matrix")

def qr_solve(A, b):
    Q, R = np.linalg.qr(A)
    y = Q.T @ b 
    return backward_substitution(R, y)

def svd_solve(A, b):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt.T @ ((U.T @ b) / s)

def cg_solve(A, b, x0, preconditioning=True, preconditioner=None, disp_full=False, conv_tol=1e-5, max_iter=100):
    res = cg.cg_solve(A, b, x0, preconditioning=preconditioning, preconditioner=preconditioner, conv_tol=conv_tol, max_iter=max_iter)
    if disp_full:
        return res 
    return res.x

LINEAR_LSTSQ_METHODS = {
    'cholesky': cholesky_solve,
    'Cholesky': cholesky_solve,
    'qr': qr_solve,
    'QR': qr_solve,
    'svd': svd_solve,
    'SVD': svd_solve,
    'cg': cg_solve,
    'CG': cg_solve
}

NON_ITERATIVE_SOLVERS = [cholesky_solve, qr_solve, svd_solve]

def converged(grad, conv_tol=1e-4):   
    return l2_norm(grad) <= conv_tol 