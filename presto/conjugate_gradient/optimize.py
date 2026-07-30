from functools import partial
import warnings 
import numpy as np 
from presto.linalg import * 
from presto.conjugate_gradient.conjugate_gradient import *
from presto.gradients import gradient_func
from presto.utils import timer, resolve_func, merge_args
from scipy.sparse.linalg import SuperLU
from scipy.sparse import csc_array
from presto.minimize_res import MinimizeResult

@dataclass
class CGResult(MinimizeResult):
    preconditioner: np.ndarray | None = None

@timer 
def cg_solve(A, b, x, preconditioning=True, preconditioner=None, conv_tol=1e-5, max_iter=100):
    x = np.array(x, dtype=float)
    A = csc_array(A, dtype=float)
    b = np.array(b, dtype=float)

    if preconditioning:
        return cg_solve_preconditioner(A, b, x, preconditioner=preconditioner, conv_tol=conv_tol, max_iter=max_iter)
    
    x_cur = x
    r_cur = A @ x_cur - b 
    p_cur = -r_cur 
    rr = r_cur @ r_cur
    f = lambda x: 0.5 * x @ A @ x - b @ x
    f_cur = f(x_cur)
    alpha = 0.0 
    iterations = []
    converged_flag = False
    for i in range(max_iter):
        iterations.append({'iter': i, 'step': alpha*p_cur, 'alpha': alpha, 'x': x_cur, 
                           'func': f_cur, 'grad': r_cur})
        if converged(r_cur, conv_tol):
            converged_flag = True
            break
        c = A @ p_cur 
        d = p_cur @ c
        if d <= 0:      
            warnings.warn(f'CG breakdown: pᵀAp ≤ 0. A is numerically indefinite - terminating search after {i} iterations')
            break      
        alpha = rr / d                     
        x_next = x_cur + alpha * p_cur
        r_next = r_cur + alpha * c  
        rr_next = r_next @ r_next 
        beta = rr_next / rr 
        p_cur = -r_next + beta * p_cur 
        r_cur, rr = r_next, rr_next
        x_cur = x_next
        f_cur = f_cur + alpha * (r_cur @ p_cur) + 0.5 * alpha**2 * d
    else:
        iterations.append({'iter': max_iter, 'step': alpha*p_cur, 'alpha': alpha, 'x': x_cur, 
                           'func': f_cur, 'grad': r_cur})

    return CGResult(
        x=x_cur,
        f_min=f_cur,
        iterations=iterations,
        func=None,
        solver='CG Fletcher Reeves',
        method='exact line search',
        converged=converged_flag,
        preconditioner=None
    )


def cg_solve_preconditioner(A, b, x, preconditioner=None, conv_tol=1e-5, max_iter=100):
    x = np.array(x, dtype=float)
    A = csc_array(A, dtype=float)
    b = np.array(b, dtype=float)

    x_cur = x
    r_cur = A @ x_cur - b 
    if preconditioner is None:
        preconditioner = incomplete_cholesky
    C = preconditioner(A) 
    y, M = factorised_solver(C)
    if isinstance(C, SuperLU):
        D = C.U.diagonal()
        if np.any(D <= 0):
            # incomplete Cholesky broke down -> A is not numerically PD
            G = None
        else:
            G = (C.L @ diags(np.sqrt(D))).toarray()
    else:
        G = M

    y_cur = y(r_cur)
    p_cur = - y_cur
    ry = r_cur @ y_cur
    f = lambda x: 0.5 * x @ A @ x - b @ x
    f_cur = f(x_cur)
    alpha = 0.0 
    converged_flag = False
    iterations = []
    for i in range(max_iter):
        iterations.append({'iter': i, 'step': alpha*p_cur, 'alpha': alpha, 'x': x_cur, 
                           'func': f_cur, 'grad': r_cur})
        if converged(r_cur, conv_tol):
            converged_flag=True
            break
        c = A @ p_cur 
        d = p_cur @ c
        if d <= 0:      
            warnings.warn(f'CG breakdown: pᵀAp ≤ 0. A is numerically indefinite - terminating search after {i} iterations')
            break 
        alpha = ry / d      
        x_next = x_cur + alpha * p_cur
        r_next = r_cur + alpha * c 
        y_next = y(r_next)
        ry_next = r_next @ y_next
        beta = ry_next / ry
        p_cur = -y_next + beta * p_cur 
        r_cur, y_cur, ry = r_next, y_next, ry_next
        x_cur = x_next
        f_cur = f_cur + alpha * (r_cur @ p_cur) + 0.5 * alpha**2 * d
    else:
        iterations.append({'iter': max_iter, 'step': alpha*p_cur, 'alpha': alpha, 'x': x_cur, 
                           'func': f_cur, 'grad': r_cur})

    return CGResult(
        x=x_cur,
        f_min=f_cur,
        iterations=iterations,
        func=None,
        solver='CG Preconditioned Fletcher Reeves',
        method='exact line search',
        converged=converged_flag,
        preconditioner=G
    )

@timer
def minimize(func, x, 
             line_search_method=None,
             conjugate_method=polak_ribiere,
             grad=None, hess=None, func_args = None, 
             line_search_args = None,
             restart=True,
             cg_restart_tol=1e-4,
             conv_tol=1e-6, max_iter=100):

    func_args = func_args or {}
    line_search_args = line_search_args or {} 
    gradient = gradient_func(func, grad, **func_args)    

    f = partial(func, **func_args)
    g = partial(gradient, **func_args) if not isinstance(gradient, partial) else gradient 
    line_search = partial(line_search_method, func, **merge_args(func_args, line_search_args))
    conjugate_beta = resolve_func(conjugate_method, NONLINEAR_CG_BETAS, 'conjugate gradient methods', polak_ribiere)

    x_cur = np.asarray(x, dtype=float)
    f_cur = f(x_cur)
    g_cur = g(x_cur)
    p_cur = - g_cur
    alpha, resets = 0.0, 0
    n = x_cur.size
    fixed_restart = True if restart and n > 20 else False
    restart_freq = n # fixed restart

    iterations = []
    converged_flag = False
    for i in range(max_iter):
        iterations.append({'iter': i, 'step': alpha*p_cur, 'alpha': alpha, 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur, 'resets': resets})
        if converged(g_cur, conv_tol):
            converged_flag = True
            break 
        alpha, x_next, f_cur = line_search(x_cur=x_cur, f_cur=f_cur, g_cur=g_cur, p_cur=p_cur)
        g_next = g(x_next)
        if fixed_restart and ((i + 1) % restart_freq == 0):
            beta = 0.0
        else:
            beta = conjugate_beta(g_cur, g_next, p_cur)
            beta = pos_beta(beta)

        p_next = - g_next + beta * p_cur 
        if restart and p_next @ g_next > -cg_restart_tol * l2_norm(p_next) * l2_norm(g_next):
            p_next = -g_next
            resets += 1
        
        g_cur = g_next
        p_cur = p_next
        x_cur = x_next

    else:
        iterations.append({'iter': max_iter, 'step': alpha*p_cur, 'alpha': alpha, 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur, 'resets': resets})

    return CGResult(
        x=x_cur,
        f_min=f_cur,
        iterations=iterations,
        func=func,
        solver=f'Nonlinear CG {conjugate_method}',
        method=line_search_method,
        converged=converged_flag,
        preconditioner=None
    )








