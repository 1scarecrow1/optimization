from functools import partial
import numpy as np
from presto.linalg import l2_norm
from presto.gradients import gradient_func, hessian_func
from presto.line_search.step_length import interpolate, cubic_interpolate, solve_cubic_interp_one


def backtracking(func, x_cur, f_cur, g_cur, p_cur, a0, c1 = 1e-4, rho=0.5, max_iter=50, a_tol=1e-10, grad=None, **func_args):
    if a0 < 0:
        raise ValueError("step length must be positive")
    if not (0 < c1 < 1):
        raise ValueError("sufficient decrease param must be between 0 and 1")
    if not (0 < rho < 1):
        raise ValueError("step length contraction factor must be between 0 and 1")
    
    alpha = a0
    f = partial(func, **func_args) if not isinstance(func, partial) else func
    x_next = x_cur + alpha * p_cur
    f_next = f(x_next) 
    suff_decrease = partial(sufficient_decrease, f_cur, g_cur=g_cur, p_cur=p_cur, c1=c1)

    for _ in range(max_iter):
        if suff_decrease(f_next=f_next, alpha=alpha):
            return alpha, x_next, f_next
        alpha *= rho
        if alpha <= a_tol:
            break 
        x_next = x_cur + alpha * p_cur
        f_next = f(x_next)
    
    raise ValueError(f'Armijo backtracking line search failed to satisfy sufficient decrease condition in {max_iter} trials')

def wolfe(func, x_cur, f_cur, g_cur, p_cur, a0=1.0, a_prev=0.0, a_max=10.0, c1=1e-4, c2=0.9, 
          max_iter=10, zoom_iter=10, interp_method="cubic", grad=None, **func_args):

    if a0 < 0 or a_prev < 0 or a_max < 0:
        raise ValueError("step length must be positive")
    if a_max < a0 or a_max < a_prev:
        raise ValueError("max step length must be larger than starting step lengths")
    if not ((0 < c1 < 1) and (0 < c2 < 1)):
        raise ValueError("sufficient decrease and curvature parameters must be between 0 and 1")
    if c2 < c1:
        raise ValueError("curvature parameter must be larger than sufficient decrease parameter")
   
    f = partial(func, **func_args) if not isinstance(func, partial) else func
    gradient = gradient_func(f, grad, **func_args) 
    g = partial(gradient, **func_args) if not isinstance(gradient, partial) else gradient 
    suff_decrease = partial(sufficient_decrease, f_cur, g_cur=g_cur, p_cur=p_cur, c1=c1)
    suff_curvature = partial(curvature, g_cur, p_cur=p_cur, c2=c2)
    zoom = partial(bracket_zoom, f, g, x_cur, p_cur, suff_decrease, suff_curvature, max_iter=zoom_iter, interp_method=interp_method)
    a = a0
    f_prev = f_cur 
    for i in range(max_iter):
        x_next = x_cur + a*p_cur
        f_next = f(x_next)
        if (not suff_decrease(f_next, alpha=a)) or (f_next >= f_prev and i > 0):
            return zoom(a_prev, a)
        g_next = g(x_next)
        if suff_curvature(g_next):
            return a, x_next, f_next
        if g_next.T @ p_cur >= 0:
            return zoom(a, a_prev)
        a_prev = a
        f_prev = f_next
        a_next = min(2*a, a_max)
        if np.isclose(a_next, a):
            break 
        a = a_next 

    raise ValueError(f"Wolfe line search failed to satisfy strong Wolfe conditions in {max_iter} trials")

def is_converged(gradient, eps=1e-4):  
    return np.isclose(l2_norm(gradient), 0.0, atol=eps) 

def bracket_zoom(f, g, x_cur, p_cur, suff_decrease, suff_curvature, a_lo, a_hi, max_iter, interp_method="cubic"): 
    phi = lambda a: f(x_cur + a*p_cur)
    phi_g = lambda a: g(x_cur + a*p_cur) @ p_cur
    phi_lo, phi_hi = phi(a_lo), phi(a_hi)
    for _ in range(max_iter):
        a = interpolate(a_lo, a_hi, phi, phi_g, phi_lo, phi_hi, method=interp_method)
        phi_a = phi(a)
        if (not suff_decrease(f_next=phi_a, alpha=a)) or phi_a >= phi_lo:
            a_hi, phi_hi = a, phi_a 
        else:
            x_next = x_cur + a*p_cur
            g_next = g(x_next)
            if suff_curvature(g_next=g_next):
                return a, x_next, phi_a
            if (g_next @ p_cur) * (a_hi - a_lo) >= 0:
                a_hi, phi_hi = a_lo, phi_lo
            a_lo, phi_lo = a, phi_a 

    raise ValueError("zoom failed to satisfy strong Wolfe conditions")

def wolfe_cond(f_next, f_cur, g_next, g_cur, p_cur, alpha, c1, c2):
    return (
        sufficient_decrease(f_cur=f_cur, f_next=f_next, g_cur=g_cur, p_cur=p_cur, alpha=alpha, c1=c1) and
        curvature(g_cur=g_cur, g_next=g_next, p_cur=p_cur, c2=c2)
            )

def sufficient_decrease(f_cur, f_next, g_cur, p_cur, alpha, c1):
    if np.ndim(f_next) == 0:
        return f_next <= f_cur + c1 * alpha * g_cur @ p_cur
    return l2_norm(f_next) <= l2_norm(f_cur + c1 * alpha * g_cur @ p_cur)

def curvature(g_cur, g_next, p_cur, c2):
    return np.abs(g_next.T @ p_cur) <= - c2 * g_cur @ p_cur

   
def interp(func, x_cur, f_cur, g_cur, p_cur, a0, c = 1e-4, rho=0.5, b=1e-4, *args, **kwargs):
    if a0 < 0:
        raise ValueError("step length must be positive")
    if not (0 < c < 1):
        raise ValueError("sufficient decrease param must be between 0 and 1")
    
    f = partial(func, *args, **kwargs)
    g = partial(func.gradient, *args, **kwargs)

    x_next = x_cur + a0 * p_cur
    f_next = f(x_next) 
    steps = [a0]
    f_v = [f_next]
    suff_decrease = partial(sufficient_decrease, f_cur=f_cur, g_cur=g_cur, p_cur=p_cur, c=c)

    if suff_decrease(f_next, alpha=a0):
        return a0, x_next, f_next

    g_next = g(x_next)
    g_v = [g_next]

    a1 = - g_cur * a0**2 / (2 * (f_next - f_cur - g_cur*a0))
    x_next = x_cur + a1 * p_cur 
    f_next = f(x_next)
    steps.append(a1)
    f_v.append(f_next)

    if suff_decrease(f_next, alpha=a1):
        return a1, x_next, f_next 

    g_next = g(x_next)
    g_v.append(g_next)

    a = solve_cubic_interp_one(a0, a1, f_v[0], f_next, f_cur, g_cur)
    x_next = x_cur + a * p_cur 
    f_next = f(x_next)
    steps.append(a)
    f_v.append(f_next)

    while not suff_decrease(f_next, alpha=a):
        a = cubic_interpolate(steps[-1], steps[-2], f_v[-1], f_v[-2], g_v[-1], g_v[-2]) 
        if np.isclose(a, steps[-1]) or np.isclose(a, b*steps[-1]):
            a = steps[-1] * rho           
        x_next = x_cur + a * p_cur
        f_next = f(x_next)
        g_next = g(x_next)
        steps.append(a)
        f_v.append(f_next)
        g_v.append(g_next)

    return steps[-1]



LINE_SEARCH_METHODS = {
    'backtracking': backtracking, 
    'armijo': backtracking,
    'armijo backtracking': backtracking,
    'wolfe': wolfe,
    'strong wolfe': wolfe
    }