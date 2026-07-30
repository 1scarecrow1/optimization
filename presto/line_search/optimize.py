from functools import partial 
import numpy as np 
from presto.linalg import * 
from presto.solvers.solvers import DIRECTIONS, QUASI_NEWTON, resolve_direction_method
from presto.line_search.line_search import LINE_SEARCH_METHODS
from presto.gradients import gradient_func, hessian_func
from presto.utils import timer, resolve_func, merge_args
from dataclasses import dataclass
from collections.abc import Callable 
from presto.minimize_res import MinimizeResult

@timer
def minimize(func, x, solver, line_search_method, 
             grad=None, hess=None, 
             func_args = None, solver_args = None, line_search_args = None,
             prev_alpha=False, # step length should be separate or part of line search
             conv_tol=1e-5, max_iter=100):
    '''
    Minimizes using line search methods 
    '''
    func_args = func_args or {}
    solver_args = solver_args or {}
    line_search_args = line_search_args or {}
    a0 = line_search_args.get('a0', 1.0)
    gradient = gradient_func(func, grad, **func_args)    

    x = np.asarray(x, dtype=float)
    f = partial(func, **func_args)
    g = partial(gradient, **func_args) if not isinstance(gradient, partial) else gradient 
    p = partial(resolve_direction_method(solver), func, **merge_args(func_args, solver_args))
    line_search = partial(resolve_func(line_search_method, LINE_SEARCH_METHODS, "line search"), f, grad=g, **merge_args(func_args, line_search_args))
    direction = p.func 

    alpha = a0
    x_cur = x
    f_cur = f(x_cur) 
    if np.ndim(f_cur) > 0:
        raise ValueError(
            "line_search.minimize expects a scalar objective. "
            "For vector-valued functions, pass a scalar merit function such as "
            "lambda x: 0.5 * l2_norm(r(x))**2, or use least_squares."
        )    
    g_cur = g(x_cur) 
    if np.ndim(g_cur) != 1:
        raise ValueError("scalar minimization expects gradient shape (n,)")
    
    p_cur = p(x_cur, g_cur)
    converged = partial(check_convergence, conv_tol=conv_tol) 
    converged_flag = False
    iterations = []
    for i in range(max_iter):
        iterations.append({'iter': i, 'step': alpha*p_cur.p, 'alpha': alpha, 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur})
        if converged(g_cur):
            print(f'{direction.__name__} with {line_search.func.__name__} converged in {i} iterations')
            converged_flag=True
            break 
        # variables needed should be fed into a state and unpacked by respective line search method
        alpha, x_next, f_next = line_search(x_cur, f_cur, g_cur, p_cur.p, a0=a0) 
        g_next = g(x_next)

        if direction in QUASI_NEWTON:
            p_cur = p(x_cur, g_cur, x_next, g_next, p_cur.B)
        else:
            p_cur = p(x_next, g_next) 

        x_cur = x_next 
        f_cur = f_next 
        g_cur = g_next        
        if prev_alpha:
            a0 = alpha
    else:
        iterations.append({'iter': i, 'step': alpha*p_cur.p, 'alpha': alpha, 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur, 'search_direction': p_cur.p})

    return MinimizeResult(
        x=x_cur,
        f_min=f_cur,
        iterations=iterations,
        func=func,
        solver=solver,
        method=line_search_method,
        converged=converged_flag
    )

def check_convergence(grad, conv_tol=1e-5):
    """
    TODO: Use test_terminal_steepest_descent 
    """      
    return l2_norm(grad) < conv_tol

def test_terminal_steepest_descent(p, grad):
    grad_norm = l2_norm(grad)
    p_norm = l2_norm(p)
    grad_val = - grad / grad_norm
    angle = - grad @ p / (grad_norm * p_norm)
    converges = np.allclose(p, grad_val, rtol=1e-5)

    res = {
        "Steepest descent direction is unit negative gradient vector": converges, 
        "search direction": p,
        "Angle is minimum": angle == 1,
        "Angle": angle
    }
    if not converges:
        res["Search direction off by"] = 100 * l2_norm(p - grad_val) / p_norm

    return res

    

    








