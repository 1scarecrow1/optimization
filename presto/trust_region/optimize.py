from functools import partial 
import numpy as np 
from presto.linalg import * 
from presto.trust_region.trust_region import quadratic_model, general, trust_region_subproblem, dogleg, check_convergence, TRUST_REGION_METHODS
import presto.solvers.solvers as solver 
from presto.solvers.solvers import NEWTON_HESSIAN_UPDATES, QUASI_NEWTON_HESSIAN_UPDATES
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
    solver: Callable | str
    trust_region_method: Callable | str


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

    if precondition:
        preconditioner = preconditioner if preconditioner is not None else inexact_modified_cholesky
        L = preconditioner(h_cur)
        g_cur = np.linalg.solve(L, g_cur)
        L_inv = np.linalg.inv(L)
        h_cur = L_inv @ h_cur @ L_inv.T

    model = partial(quadratic_model, D=D)
    general_solve = partial(general, func, model, method=trust_region_method, 
                            method_args=method_args, solver_args=solver_args)

    num_iters = 0
    iterations = [{'iter': num_iters, 'step': np.zeros_like(x_cur), 'size': rad_cur, 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur}]
    converged = partial(check_convergence, conv_tol=conv_tol)
    while not (converged(g_cur) or num_iters >= max_iter or np.isclose(rad_cur, 0.0)): 
        p_cur, x_next, f_cur, rad_cur = general_solve(x_cur, f_cur, g_cur, h_cur, rad_cur)
        if not np.allclose(p_cur, 0):
            if precondition:
                p_cur = np.linalg.solve(L.T, p_cur)
            g_next = g(x_next)
            if solver in NEWTON_HESSIAN_UPDATES:
                h_cur = h(x_next, g_next)
            elif solver in QUASI_NEWTON_HESSIAN_UPDATES:
                h_cur = h(x_cur, g_cur, x_next, g_next, h_cur)
            g_cur = g_next 

        x_cur = x_next 
        num_iters += 1
        iterations.append({'iter': num_iters, 'step': p_cur, 'size': rad_cur, 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur})


    return MinimizeResult(
        x=x_cur,
        f_min=f_cur,
        iterations=iterations,
        func=func,
        solver=solver,
        trust_region_method=trust_region_method,
    )


def resolve_trust_region_method(f, methods=TRUST_REGION_METHODS, name="trust region"):
    return resolve_func(f, methods, name)
    

