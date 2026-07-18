from functools import partial 
import numpy as np 
from presto.linalg import * 
from presto.trust_region import dogleg, TRUST_REGION_METHODS
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
    size: float
    #search_direction: Callable | str
    trust_region_method: Callable | str


@timer
def minimize(func, x, 
             # search_direction,
             trust_region_method, grad=None, hess=None, 
             func_args = None, 
             #direction_args = None, 
             trust_region_args = None,
             conv_tol=1e-5, max_iter=100):
    
    func_args = func_args or {}
    #direction_args = direction_args or {}
    trust_region_args = trust_region_args or {}
    rad_cur = trust_region_args['rad0'] if trust_region_args else 1.0
    rad_max = trust_region_args['rad_max'] if trust_region_args else 2.0

    x = np.asarray(x, dtype=float)

    #direction = resolve_direction_method(search_direction)
    trust_region_func = resolve_trust_region_method(trust_region_method)
    gradient = gradient_func(func, grad, **func_args)    
    hessian = hessian_func(func, hess, **func_args)

    f = partial(func, **func_args)
    g = partial(gradient, **func_args) if not isinstance(gradient, partial) else gradient 
    h = partial(hessian, **func_args) if not isinstance(hessian, partial) else hessian
    trust_region = partial(trust_region_func, func, rad_max=rad_max, **merge_args(func_args, trust_region_args))

    x_cur = x
    rad_cur = trust_region_args['rad0'] if trust_region_args else 2.0
    f_cur = f(x_cur) 
    g_cur = g(x_cur) 
    h_cur = h(x_cur)

    num_iters = 0
    iterations = [{'iter': num_iters, 'step': np.zeros(len(x)), 'size': rad_cur, 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur}]
    converged = partial(check_convergence, conv_tol=conv_tol) 

    while not (converged(g_cur) or num_iters >= max_iter): 
        p, rad_next, x_next, f_next = trust_region(x_cur, f_cur, g_cur, h_cur, rad_cur)
        g_cur = g(x_next)
        g_next = g(x_next)

        # if direction in QUASI_NEWTON:
        #     p_cur = p(x_cur, g_cur, x_next, g_next, p_cur.B)
        # else:
        #     p_cur = p(x_cur, g_cur) 
        rad_cur = rad_next
        x_cur = x_next 
        f_cur = f_next 
        g_cur = g_next 

        num_iters += 1
        iterations.append({'iter': num_iters, 'step': p, 'size': rad_cur, 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur})


    return MinimizeResult(
        x=x_cur,
        f_min=f_cur,
        iterations=iterations,
        func=func,
        search_direction=search_direction,
        trust_region_method=trust_region_method,
    )

def check_convergence(gradient, eps=1e-4):
    """
    TODO: Use test_terminal_steepest_descent 
    """      
    return np.isclose(l2_norm(gradient), 0.0, rtol=eps) 


def resolve_trust_region_method(f, methods=TRUST_REGION_METHODS, name="trust region"):
    return resolve_func(f, methods, name)
    
# def resolve_direction_method(f, methods=DIRECTIONS, name='direction'):
#     return resolve_func(f, methods, name)







