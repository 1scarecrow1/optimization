from functools import partial
import warnings 
import numpy as np 
from presto.linalg import * 
from presto.trust_region.trust_region import quadratic_model, general, trust_region_subproblem, dogleg, check_convergence, TRUST_REGION_METHODS
import presto.solvers.solvers as solver 
from presto.solvers.solvers import NEWTON_HESSIAN_UPDATES, QUASI_NEWTON_HESSIAN_UPDATES
from presto.gradients import gradient_func, hessian_func
from presto.utils import timer, resolve_func, merge_args
from presto.minimize_res import MinimizeResult
from collections.abc import Callable 

@timer
def minimize(func, x, solver, trust_region_method=dogleg, 
             grad=None, hess=None, 
             func_args = None, solver_args=None, trust_region_args = None,
             precondition=False, preconditioner=None,
             conv_tol=1e-4, max_iter=100):
    '''
    TODO: General optimizer class/function
    '''
    func_args = func_args or {}
    solver_args = solver_args or {}
    trust_region_args = trust_region_args or {}
    trust_region_method = resolve_trust_region_method(trust_region_method)
    
    method_args = {k: v for k, v in trust_region_args.items() if k not in ('rad0', 'D')}
    D = trust_region_args.get('D', None)
    hessian_approx_args = {k: v for k, v in solver_args.items() if k in ('inv', 'scale', 'den_tol')}
    gradient = gradient_func(func, grad, **func_args)    

    f = partial(func, **func_args)
    g = partial(gradient, **func_args) if not isinstance(gradient, partial) else gradient 

    if solver in NEWTON_HESSIAN_UPDATES:
        hessian = hessian_func(func, grad, hess, **func_args)
        h = partial(hessian, f, **func_args) if not isinstance(hessian, partial) else hessian
    elif solver in QUASI_NEWTON_HESSIAN_UPDATES:
        h = partial(solver, **hessian_approx_args)
    else:
        raise NotImplementedError

    x_cur = np.array(x, dtype=float)
    rad_cur = trust_region_args.get('rad0', 100.0)
    f_cur = f(x_cur) 
    g_cur = g(x_cur) 
    h_cur = h(x_cur, g_cur)
    p_cur = np.zeros_like(x_cur) 

    if precondition:
        preconditioner = preconditioner if preconditioner is not None else inexact_modified_cholesky
        L = preconditioner(h_cur)
        g_cur = np.linalg.solve(L, g_cur)
        L_inv = np.linalg.inv(L)
        h_cur = L_inv @ h_cur @ L_inv.T

    general_solve = partial(general, func, method=trust_region_method, D_cur=D,
                            method_args=method_args, solver_args=solver_args)

    iterations = []
    converged = partial(check_convergence, conv_tol=conv_tol)
    converged_flag = False
    for i in range(max_iter):
        iterations.append({'iter': i, 'step': p_cur, 'size': rad_cur, 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur})
        if converged(g_cur):
            print(f'{solver.__name__} with {trust_region_method.__name__} converged in {i} iterations')
            converged_flag = True
            break 
        if np.isclose(rad_cur, 0.0):
            warnings.warn(f'trust region radius close to 0: {rad_cur} - terminating search')
            break 
        p_cur, x_next, f_cur, rad_cur = general_solve(x_cur, f_cur, g_cur, h_cur, rad_cur)
        if not np.allclose(p_cur, 0):
            g_next = g(x_next)
            if solver in NEWTON_HESSIAN_UPDATES:
                h_cur = h(x_next, g_next)
            elif solver in QUASI_NEWTON_HESSIAN_UPDATES:
                h_cur = h(x_cur, g_cur, x_next, g_next, h_cur)
            if precondition:
                L = preconditioner(h_cur)
                p_cur = np.linalg.solve(L.T, p_cur)
                g_next = np.linalg.solve(L, g_next)
                L_inv = np.linalg.inv(L)
                h_cur = L_inv @ h_cur @ L_inv.T
            g_cur = g_next 
        x_cur = x_next 
    else:
        iterations.append({'iter': max_iter, 'step': p_cur, 'size': rad_cur, 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur})

    return MinimizeResult(
        x=x_cur,
        f_min=f_cur,
        iterations=iterations,
        func=func,
        solver=solver,
        method=trust_region_method,
        converged=converged_flag
    )


def resolve_trust_region_method(f, methods=TRUST_REGION_METHODS, name="trust region"):
    return resolve_func(f, methods, name)
    

