from functools import partial 
import numpy as np 
from presto.linalg import * 
from presto.conjugate_gradient.conjugate_gradient import *
from presto.gradients import gradient_func
from presto.utils import timer, resolve_func, merge_args
from dataclasses import dataclass
from collections.abc import Callable 
from scipy.sparse.linalg import SuperLU
from scipy.sparse import csc_array

@dataclass
class MinimizeResult:
    x: np.ndarray
    f_min: float
    iterations: list[dict]
    func: Callable | None = None
    preconditioner: np.ndarray | None = None
    conjugate_method: str | Callable | None = None
    line_search_method: str | Callable | None = None

@timer 
def cg_solve(A, b, x, preconditioning=True, preconditioner=None, conv_tol=1e-5, max_iter=100):
    x = np.asarray(x, dtype=float)
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

    num_iters = 0
    iterations = [{'iter': num_iters, 'direction': p_cur, 'x': x_cur, 
                           'func': f_cur, 'grad': r_cur}]
    
    while not (converged(r_cur, conv_tol) or num_iters >= max_iter):
        c = A @ p_cur 
        d = p_cur @ c
        if d <= 0:                      
            print("CG breakdown: pᵀAp ≤ 0 (A is numerically indefinite)")
            break                        
        alpha = rr / d
        x_next = x_cur + alpha * p_cur
        r_next = r_cur + alpha * c  
        rr_next = r_next @ r_next 
        beta = rr_next / rr 
        p_cur = -r_next + beta * p_cur 
        r_cur = r_next
        rr = rr_next
        x_cur = x_next
        f_cur = f_cur + alpha * (r_cur @ p_cur) + 0.5 * alpha**2 * d

        num_iters += 1
        iterations.append({'iter': num_iters, 'direction': p_cur, 'x': x_cur, 
                           'func': f_cur, 'grad': r_cur})

    return MinimizeResult(
        x=x_cur,
        f_min=f_cur,
        iterations=iterations,
        func=None,
        preconditioner=None,
        conjugate_method=fletcher_reeves,
        line_search_method='exact'
    )


def cg_solve_preconditioner(A, b, x, preconditioner=None, conv_tol=1e-5, max_iter=100):
    x = np.asarray(x, dtype=float)
    A = np.array(A, dtype=float)
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

    num_iters = 0
    iterations = [{'iter': num_iters, 'direction': p_cur, 'x': x_cur, 
                           'func': f_cur, 'grad': r_cur}]
    
    while not (converged(r_cur, conv_tol) or num_iters >= max_iter):
        c = A @ p_cur 
        d = p_cur @ c
        if d <= 0:                      
            print("CG breakdown: pᵀAp ≤ 0 (A is numerically indefinite)")
            break    
        alpha = ry / d
        x_next = x_cur + alpha * p_cur
        r_next = r_cur + alpha * c 
        y_next = y(r_next)
        ry_next = r_next @ y_next
        beta = ry_next / ry
        p_cur = -y_next + beta * p_cur 
        r_cur = r_next
        y_cur = y_next
        ry = ry_next 
        x_cur = x_next
        f_cur = f_cur + alpha * (r_cur @ p_cur) + 0.5 * alpha**2 * d

        num_iters += 1
        iterations.append({'iter': num_iters, 'direction': p_cur, 'x': x_cur, 
                           'func': f_cur, 'grad': r_cur})

    return MinimizeResult(
        x=x_cur,
        f_min=f_cur,
        iterations=iterations,
        func=None,
        preconditioner=G,
        conjugate_method='preconditioned_fletcher_reeves',
        line_search_method='exact'
    )

@timer
def minimize(func, x, 
             line_search_method=None,
             conjugate_method=polak_ribiere,
             grad=None, hess=None, func_args = None, 
             line_search_args = None,
             conv_tol=1e-5, max_iter=100):
    '''
    TODO: General optimizer class/function
    '''
    func_args = func_args or {}
    line_search_args = line_search_args or {} 

    x = np.asarray(x, dtype=float)

    gradient = gradient_func(func, grad, **func_args)    

    f = partial(func, **func_args)
    g = partial(gradient, **func_args) if not isinstance(gradient, partial) else gradient 
    line_search = partial(line_search_method, func, **merge_args(func_args, line_search_args))
    
    x_cur = x
    f_cur = f(x_cur)
    g_cur = g(x_cur)
    p_cur = - g_cur
    num_iters = 0
    iterations = [{'iter': num_iters, 'direction': p_cur, 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur}]
    
    while not (converged(g_cur, conv_tol) or num_iters >= max_iter):
        _, x_next, f_cur = line_search(x_cur=x_cur, f_cur=f_cur, g_cur=g_cur, p_cur=p_cur)
        g_next = g(x_next)
        beta = conjugate_method(g_cur, g_next, p_cur)
        g_cur = g_next
        p_cur = - g_cur + beta * p_cur 
        x_cur = x_next

        num_iters += 1
        iterations.append({'iter': num_iters, 'direction': p_cur, 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur})

    return MinimizeResult(
        x=x_cur,
        f_min=f_cur,
        iterations=iterations,
        func=func,
        preconditioner=None,
        conjugate_method=conjugate_method,
        line_search_method=line_search_method
    )








