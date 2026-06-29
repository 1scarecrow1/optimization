from functools import partial
import numpy as np

def backtracking(func, x_cur, f_cur, g_cur, p_cur, a0, c = 1e-4, rho=0.5, *args, **kwargs):
    if a0 < 0:
        raise ValueError("step length must be positive")
    if not (0 < c < 1):
        raise ValueError("sufficient decrease param must be between 0 and 1")
    if not (0 < rho < 1):
        raise ValueError("step length contraction factor must be between 0 and 1")
    
    alpha = a0
    f = partial(func, *args, **kwargs)  
    x_next = x_cur + alpha * p_cur
    f_next = f(x_next) 

    while f_next > f_cur + c * alpha * g_cur @ p_cur:
        alpha *= rho
        x_next = x_cur + alpha * p_cur
        f_next = f(x_next)

    return alpha, x_next, f_next


def wolfe():
    raise NotImplementedError