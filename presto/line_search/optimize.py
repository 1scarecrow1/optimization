from functools import partial 
import numpy as np 
from presto.linalg import * 
from presto.line_search.direction import DIRECTIONS, QUASI_NEWTON, SECOND_ORDER_DIRECTIONS
from presto.line_search.line_search import LINE_SEARCH_METHODS
from presto.gradients import gradient_func, hessian_func
from presto.utils import timer, resolve_func, merge_args
from dataclasses import dataclass
from collections.abc import Callable 

@dataclass
class MinimizeResult:
    x: np.ndarray
    f_min: float
    iterations: list[dict]
    func: Callable
    search_direction: Callable | str
    line_search_method: Callable | str


@timer
def minimize(func, x, search_direction, line_search_method, 
             grad=None, hess=None, 
             func_args = None, direction_args = None, line_search_args = None,
             prev_alpha=False, # step length should be separate or part of line search
             conv_tol=1e-5, max_iter=100):
    '''
    Minimizes using line search methods and directions
    '''
    func_args = func_args or {}
    direction_args = direction_args or {}
    line_search_args = line_search_args or {}
    a0 = line_search_args['a0'] if line_search_args else 1.0

    x = np.asarray(x, dtype=float)

    direction = resolve_direction_method(search_direction)
    line_search_func = resolve_line_search_method(line_search_method)
    gradient = gradient_func(func, grad, **func_args)    

    f = partial(func, **func_args)
    g = partial(gradient, **func_args) if not isinstance(gradient, partial) else gradient 
    p = partial(direction, func, **merge_args(func_args, direction_args))
    line_search = partial(line_search_func, func, **merge_args(func_args, line_search_args))

    if direction in SECOND_ORDER_DIRECTIONS:
        hessian = hessian_func(func, hess, **func_args)
        h = partial(hessian, **func_args) if not isinstance(hessian, partial) else hessian
    
    alpha = a0
    x_cur = x
    f_cur = f(x_cur) 
    g_cur = g(x_cur) 
    p_cur = p(x_cur, g_cur)

    num_iters = 0
    iterations = [{'iter': num_iters, 'step': 0, 'x': x_cur, 'func': f_cur, 'grad': g_cur, 'search_direction': p_cur.p}]
    converged = partial(check_convergence, conv_tol=conv_tol) 

    while not (converged(g_cur) or num_iters >= max_iter): 
        # variables needed should be fed into a state and unpacked by respective line search method
        alpha, x_next, f_next = line_search(x_cur, f_cur, g_cur, p_cur.p, a0=a0)
        g_next = g(x_next)

        if direction in QUASI_NEWTON:
            p_cur = p(x_cur, g_cur, x_next, g_next, p_cur.B)
        else:
            p_cur = p(x_cur, g_cur) 

        x_cur = x_next 
        f_cur = f_next 
        g_cur = g_next 

        num_iters += 1
        iterations.append({'iter': num_iters, 'step': alpha, 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur, 'search_direction': p_cur.p})
        
        if prev_alpha:
            a0 = alpha

    return MinimizeResult(
        x=x_cur,
        f_min=f_cur,
        iterations=iterations,
        func=func,
        search_direction=search_direction,
        line_search_method=line_search_method,
    )

def check_convergence(grad, conv_tol=1e-4):
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

    
def resolve_line_search_method(f, methods=LINE_SEARCH_METHODS, name="line search"):
    return resolve_func(f, methods, name)
    
def resolve_direction_method(f, methods=DIRECTIONS, name='direction'):
    return resolve_func(f, methods, name)







