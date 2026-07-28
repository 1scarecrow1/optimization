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

def cubic_interpolate(a1, a0, f1, f0, g1, g0):
    d1 = g0 + g1 - 3*(f0 - f1) / (a1 - a0)
    d2 = np.sign(a1 - a0)* np.sqrt(d1**2 - g0*g1)
    return a0 * (g1 + d2 - d1) / (g1 - g0 + 2*d2)

def interpolate(f, a_lo, a_hi, f1=None, f0=None, g1=None, g0=None, method='bisection'):

    if method=='bisection':
        return bisection(f, a_lo, a_hi)
    
    if method=='quadratic':
        return quadratic_interpolate(f1, f0, g1)

    if method=='cubic':
        return cubic_interpolate(a_lo, a_hi, f1, f0, g1, g0)   

def bisection(f, a, b):
    ''''
    TODO: Add termination criteria
    '''
    c = (a + b) / 2
    mid = f(c)
    if np.isclose(mid, 0):
        return c 
    
    while not np.isclose(mid, 0):
        if np.sign(mid) != np.sign(l2_norm(f(a))):
            c = (a+c) / 2
            mid = f(c)
        else:
            c = (c+b) / 2
            mid = f(c)
    return c 

   
def constant_first_order_change(alpha_prev, g_prev, p_prev, g_cur, p_cur):
    return alpha_prev * (g_prev.T @ p_prev) / (g_cur.T @ p_cur)  

def quadratic_interpolate(f_cur, f_prev, g_cur):
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



INITIAL_STEP_LENGTH = {
    'constant_fo_change': constant_first_order_change, 
    'quadratic_interpolation': quadratic_interpolate,
    'cubic_interpolation': cubic_interpolate,
    'bisection': bisection
    }