from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import warnings
import numpy as np
from presto.conjugate_gradient.conjugate_gradient import converged, fletcher_reeves
from presto.gradients import gradient_func, hessian_func
from presto.linalg import inexact_modified_cholesky, l2_norm, inverse
from presto.trust_region.newton_cg import cg_steihaug
from presto.utils import merge_args

@dataclass
class MinimizeResult:
    x: np.ndarray
    f_min: float
    iterations: list[dict]
    func: Callable | None = None
    preconditioner: np.ndarray | None = None
    line_search_method: str | Callable | None = None

def minimize(func, x, 
             line_search=True,
             line_search_method=None,
             grad=None, hess=None, func_args = None, 
             line_search_args = None,
             preconditioner=None,
             conv_tol=1e-5, max_iter=100):

    func_args = func_args or {}
    line_search_args = line_search_args or {} 
    gradient = gradient_func(func, grad, **func_args)    
    hessian = hessian_func(func, grad, hess, **func_args)
    
    f = partial(func, **func_args)
    g = partial(gradient, **func_args) if not isinstance(gradient, partial) else gradient 
    h = partial(hessian, **func_args) if not isinstance(hessian, partial) else hessian
    if line_search:
        line_search_func = partial(line_search_method, func, a0=1, **merge_args(func_args, line_search_args))
    else:
        rad_cur = 2
    
    x_cur = np.asarray(x, dtype=float)
    f_cur = f(x_cur)
    g_cur = g(x_cur)
    h_cur = h(x_cur)

    if not line_search:
        precondition = preconditioner if preconditioner is not None else inexact_modified_cholesky
        L = precondition(h_cur)
        g_cur = np.linalg.solve(L, g_cur)
        L_inv = inverse(L)
        h_cur = L_inv @ h_cur @ L_inv.T

    g_cur_norm = l2_norm(g_cur)
    tol_cur = min(0.5, np.sqrt(g_cur_norm)) * g_cur_norm

    num_iters = 0
    iterations = [{'iter': num_iters, 'direction': np.zeros_like(x_cur), 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur}]
    
    #while g_cur_norm > conv_tol or num_iters < max_iter:
    for i in range(1, max_iter+1):
        if line_search:
            p_cur = cg_line_search(g_cur, h_cur, tol_cur)
            _, x_cur, f_cur = line_search_func(x_cur=x_cur, f_cur=f_cur, g_cur=g_cur, p_cur=p_cur)
        else:
            p_cur = cg_steihaug(g_cur, h_cur, rad_cur, tol_cur=tol_cur)
            p_cur = np.linalg.solve(L.T, p_cur)      # p = L⁻ᵀ p̂
            x_cur = x_cur + p_cur 
            f_cur = f(x_cur)
        g_cur = g(x_cur)
        iterations.append({'iter': i, 'direction': p_cur, 'x': x_cur, 
                           'func': f_cur, 'grad': g_cur})
        g_cur_norm = l2_norm(g_cur)
        if g_cur_norm < conv_tol:
            break
        tol_cur = min(0.5, np.sqrt(g_cur_norm)) * g_cur_norm
        h_cur = h(x_cur)

        if not line_search:
            L = precondition(h_cur)
            g_cur = np.linalg.solve(L, g_cur)
            L_inv = np.linalg.inv(L)
            h_cur = L_inv @ h_cur @ L_inv.T

    return MinimizeResult(
        x=x_cur,
        f_min=f_cur,
        iterations=iterations,
        func=func,
        preconditioner=preconditioner,
        line_search_method=line_search_method
    )

    
def cg_line_search(g_cur, B_cur, tol_cur, max_iter=100):
    z_cur = np.zeros_like(g_cur)
    r_cur = g_cur
    d_cur = -r_cur 
    c = B_cur @ d_cur
    if d_cur @ c <= 0:
        return -g_cur
    rr = r_cur @ r_cur

    for _ in range(max_iter):
        curvature = d_cur @ c
        if curvature <= 0:
            return z_cur

        alpha = rr / curvature
        z_cur = z_cur + alpha * d_cur
        r_next = r_cur + alpha * c  
        rr_next = r_next @ r_next
        if l2_norm(r_next) < tol_cur:
            return z_cur 
        beta = rr_next / rr 
        d_cur = -r_next + beta*d_cur 
        c = B_cur @ d_cur
        r_cur = r_next
        rr = rr_next
    warnings.warn(f'newton cg failed to converge after {max_iter} iterations', RuntimeWarning)
    return z_cur   




