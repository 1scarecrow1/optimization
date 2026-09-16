import numpy as np  
from functools import partial 
from presto.linalg import l2_norm

def step_length_func(func, alpha, x_cur, p_cur, **kwargs):
    f = partial(func, **kwargs)
    return f(x_cur + alpha * p_cur)

def step_length_func_gradient(func, alpha, x_cur, p_cur, **kwargs):
    f = partial(func, **kwargs)
    g = partial(func.gradient, **kwargs)
    return g(x_cur + alpha*p_cur).T @ p_cur

step_length_func.gradient = step_length_func_gradient

def quadratic_approx(alpha, a0, f_0, f_cur, g_cur):
    return ((f_0 - f_cur - a0*g_cur)/a0**2)*alpha**2 + g_cur*alpha + f_cur 

def solve_cubic_interp_one(a0, a1, f_0, f_1, f_cur, g_cur):
    c = 1 / (a0**2 * a1**2 * (a1 - a0))
    s = np.array([a1, a0])
    f_v = np.array([f_1, f_0])
    A = np.array([[a0**2, -a1**2], [-a0**3, a1**3]])
    v = f_v - g_cur*s - f_cur
    x = c * A @ v 
    a, b = x 
    return (-b + np.sqrt(b**2 - 3*a*g_cur)) / (3*a)

def quadratic_interpolate(a_lo, a_hi, phi, phi_g, phi_lo=None, phi_hi=None):
    if phi_lo is None:
        phi_lo = phi(a_lo)
    if phi_hi is None:
        phi_hi = phi(a_hi)
    phi_g_lo = phi_g(a_lo)

    da = a_hi - a_lo
    denom = 2 * (phi_hi - phi_lo - phi_g_lo * da)

    if np.isclose(denom, 0.0):
        return bisection(a_lo, a_hi)

    a = a_lo - phi_g_lo * da**2 / denom
    return safeguard_alpha(a, a_lo, a_hi)

def safeguard_alpha(a, a_lo, a_hi, frac=0.1):
    lo, hi = sorted((a_lo, a_hi))
    width = hi - lo
    lower = lo + frac * width
    upper = hi - frac * width

    if not np.isfinite(a):
        return bisection(a_lo, a_hi)

    return min(max(a, lower), upper)

def cubic_interpolate(a_lo, a_hi, phi, phi_g, phi_lo=None, phi_hi=None):
    if phi_lo is None:
        phi_lo = phi(a_lo)
    if phi_hi is None:
        phi_hi = phi(a_hi)
    phi_g_lo = phi_g(a_lo)
    phi_g_hi = phi_g(a_hi)

    d1 = phi_g_lo + phi_g_hi - 3 * (phi_lo - phi_hi) / (a_lo - a_hi)
    disc = d1**2 - phi_g_lo * phi_g_hi

    if disc < 0:
        return bisection(a_lo, a_hi)

    d2 = np.sign(a_hi - a_lo) * np.sqrt(disc)
    denom = phi_g_hi - phi_g_lo + 2 * d2

    if np.isclose(denom, 0.0):
        return bisection(a_lo, a_hi)

    a = a_hi - (a_hi - a_lo) * (phi_g_hi + d2 - d1) / denom
    return safeguard_alpha(a, a_lo, a_hi)

def interpolate(a_lo, a_hi, phi=None, phi_g=None, phi_lo=None, phi_hi=None, method=None):
    method = method or 'cubic'

    if method=='bisection' or phi is None or phi_g is None:
        return bisection(a_lo, a_hi)
   
    if method=='quadratic':
        return quadratic_interpolate(a_lo, a_hi, phi, phi_g, phi_lo=phi_lo, phi_hi=phi_hi)

    if method=='cubic':
        return cubic_interpolate(a_lo, a_hi, phi, phi_g, phi_lo=phi_lo, phi_hi=phi_hi)   
    
    raise ValueError(f"unknown interpolation method: {method}")

def bisection(a_lo, a_hi):
    return 0.5*(a_lo + a_hi)

def bisection2(f, a, b, max_iter):
    ''''
    TODO: Add termination criteria
    '''
    c = (a + b) / 2
    mid = f(c)
    for _ in range(max_iter):
        if np.isclose(l2_norm(mid), 0):
            return c         

        if np.sign(mid) != np.sign(f(a)):
            c = (a+c) / 2
        else:
            c = (c+b) / 2
        mid = f(c)
    print(f'bisection failed to converge in {max_iter} iterations')
    return c 

def constant_first_order_change(alpha_prev, g_prev, p_prev, g_cur, p_cur):
    return alpha_prev * (g_prev.T @ p_prev) / (g_cur.T @ p_cur)  

def quadratic_initial_step(f_cur, f_prev, g_cur):
    return 2*(f_cur - f_prev) / g_cur

def initial_step_length_newton():
    return 1.0 

def initial_step_length_gradient_desc(alpha_prev=None, g_prev=None, p_prev=None, g_cur=None, p_cur=None, f_cur=None, f_prev=None):
    constant_fo_args = (alpha_prev, g_prev, p_prev, g_cur, p_cur)
    quad_interp_args = (f_cur, f_prev, g_cur)

    if alpha_prev is not None:
        if all(x is not None for x in constant_fo_args):
            return constant_first_order_change(alpha_prev, g_prev, p_prev, g_cur, p_cur)
        else:
            return alpha_prev
        
    if all(x is not None for x in quad_interp_args):
        return quadratic_interpolate(f_cur, f_prev, g_cur)

    return 1.0

# def initial_step_length(search_direction=None, alpha_prev=None, f_cur=None, f_prev=None, g_prev=None, p_prev=None, g_cur=None, p_cur=None):
#     if search_direction is not None:
#         if search_direction in NEWTON or QUASI_NEWTON:
#             return initial_step_length_newton()
#         if search_direction in GRADIENT_DESCENT or CONJUGATE_GRADIENT:
#             return initial_step_length_gradient_desc(
#                 alpha_prev=alpha_prev, g_prev=g_prev, p_prev=p_prev, g_cur=g_cur, p_cur=p_cur, f_cur=f_cur, f_prev=f_prev)
        
#     elif alpha_prev is not None:
#         return alpha_prev 
    
#     else:
#         return 1.0


INITIAL_STEP_LENGTH = {
    'constant_fo_change': constant_first_order_change, 
    'quadratic_interpolation': quadratic_interpolate,
    'cubic_interpolation': cubic_interpolate,
    'bisection': bisection
    }