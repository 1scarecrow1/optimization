from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import logging
import warnings
import numpy as np
from presto.gradients import gradient_func
from presto.linalg import l2_norm
from presto.least_squares.linear import LINEAR_LSTSQ_METHODS, qr_solve, NON_ITERATIVE_SOLVERS
from presto.least_squares.loss_functions import LOSS_FUNCTIONS, quadratic_loss
from presto.line_search.line_search import backtracking, wolfe
from presto.trust_region.trust_region import general, gauss_newton_model, qr_trust_region_subproblem
from presto.utils import resolve_func

logger = logging.getLogger(__name__)

###### Function supplied should be least squares residual 
###### Objective function is scalar merit function - default is squared loss

@dataclass
class MinimizeResult:
    x: np.ndarray
    f_min: np.ndarray | float
    iterations: list[dict]
    func: Callable
    solver: Callable | str
    search_method: Callable | str

def gauss_newton(func, x0, jac=None, loss_function='sq',
                 subproblem_method=qr_solve, subproblem_args=None, 
                 line_search_method=wolfe, line_search_args=None,
                 loss_function_args = None,
                 max_iter=200, **func_args):

    func_args = func_args or {}
    loss_function_args = loss_function_args or {}
    r = partial(func, **func_args) 
    f = partial(resolve_func(loss_function, LOSS_FUNCTIONS, "loss function", quadratic_loss), r, **loss_function_args)
    jacobian = gradient_func(r, jac)    
    subproblem_args = subproblem_args or {}
    solve_lstsq = partial(resolve_func(subproblem_method, LINEAR_LSTSQ_METHODS, "linear least squares", default=qr_solve), **subproblem_args)
    linear_lstsq_method = solve_lstsq.func

    x_cur = np.asarray(x0, dtype=float)
    J = partial(jacobian, **func_args) if not isinstance(jacobian, partial) else jacobian 
    f_cur = f(x_cur)
    r_cur = r(x_cur)
    j_cur = J(x_cur)
    g_cur = j_cur.T @ r_cur 
    p_cur = np.zeros_like(x_cur)

    line_search_args = line_search_args or {}
    line_search_args['a0'] = 1.0 # overwrite any starting alpha values with 1
    alpha = line_search_args['a0']
    line_search = partial(line_search_method, f, **line_search_args)

    iterations = []
    for i in range(max_iter):
        iterations.append({'iter': i, 'step': alpha, 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur, 'search_direction': p_cur})
        if converged(g_cur):
            break
        if linear_lstsq_method in NON_ITERATIVE_SOLVERS: 
            p_cur = solve_lstsq(j_cur, -r_cur)
        else:
            p_cur = solve_lstsq(j_cur, -r_cur, x_cur) 
        alpha, x_cur, f_cur = line_search(x_cur, f_cur, g_cur, p_cur)
        r_cur = r(x_cur)
        j_cur = J(x_cur)
        g_cur = j_cur.T @ r_cur         

    else:
        iterations.append({'iter': max_iter, 'step': alpha, 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur, 'search_direction': p_cur})

    return MinimizeResult(
        x=x_cur,
        f_min=f_cur,
        iterations=iterations,
        func=func,
        solver='gauss_newton',
        search_method=line_search_method,
    )

def levenberg_marquardt(func, x0, jac=None, loss_function='sq',
                 subproblem_method=None, subproblem_args=None, 
                 trust_region_method=qr_trust_region_subproblem, trust_region_args=None,
                 loss_function_args = None,
                 max_iter=200, **func_args):

    func_args = func_args or {}
    loss_function_args = loss_function_args or {}
    trust_region_args = trust_region_args or {}
    method_args = {k: v for k, v in trust_region_args.items() if k not in ('rad0', 'D')}

    r = partial(func, **func_args) 
    f = partial(resolve_func(loss_function, LOSS_FUNCTIONS, "loss function", quadratic_loss), r, **loss_function_args)
    jacobian = gradient_func(r, jac)    

    x_cur = np.asarray(x0, dtype=float)
    J = partial(jacobian, **func_args) if not isinstance(jacobian, partial) else jacobian 
    f_cur = f(x_cur)
    r_cur = r(x_cur)
    j_cur = J(x_cur)
    g_cur = j_cur.T @ r_cur 
    rad_cur = trust_region_args.get('rad0', 100.0)
    p_cur = np.zeros_like(x_cur)
    D = trust_region_args.get('D', None)
    
    general_solve = partial(general, f, method=trust_region_method, model=gauss_newton_model, D_cur=D,
                            method_args=method_args)
    iterations = []
    for i in range(max_iter):
        iterations.append({'iter': i, 'step': p_cur, 'size': rad_cur, 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur})
        if np.isclose(rad_cur, 0.0):
            warnings.warn(f'trust region radius close to 0: {rad_cur} - terminating search')
            break 
        if converged(g_cur):
            logger.info(f'levenberg marquardt with {trust_region_method.__name__} converged in {i} iterations')
            break        
        p_cur, x_cur, f_cur, rad_cur = general_solve(x_cur, f_cur, r_cur, j_cur, rad_cur)
        r_cur = r(x_cur)
        j_cur = J(x_cur)
        g_cur = j_cur.T @ r_cur 
    else:
        iterations.append({'iter': max_iter, 'step': p_cur, 'size': rad_cur, 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur})
    return MinimizeResult(
        x=x_cur,
        f_min=f_cur,
        iterations=iterations,
        func=func,
        solver='levenberg_marquardt',
        search_method=trust_region_method,
    )
        
def converged(grad, conv_tol=1e-4):   
    return l2_norm(grad) <= conv_tol 

NONLINEAR_LSTSQ_METHODS = {
    'gauss_newton': gauss_newton,
    'gn': gauss_newton,
    'lm': levenberg_marquardt,
    'levenberg_marquardt': levenberg_marquardt
}

