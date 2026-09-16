import logging
import warnings
import numpy as np
from presto.gradients import gradient, hessian
from presto.linalg import l2_norm, inverse

logger = logging.getLogger(__name__)
machine_eps = np.finfo(np.float32).eps

def minimize(grad, x, hess, max_iter=100, tol=1e-4, eps=machine_eps):
    '''
    Solve f'(x) = 0
    '''
    x = roots(grad, x, hess, max_iter=max_iter, tol=tol, eps=eps)
    return x  

def roots(func, x, grad, max_iter=200, tol=1e-4, eps=machine_eps):
    '''
    Solve f(x) = 0 or f'(x) = 0
    For f'(x) = 0, func is grad and grad is hess
    '''
    x = np.array(x)
    f_cur = func(x)
    g_cur = grad(x)

    funcevals = 1
    gradevals = 1
    step = newton_iteration(f_cur, g_cur)
    x_next = x + step

    for _ in range(max_iter):
        if stopping_criteria(g_cur, atol=eps):
            raise ZeroDivisionError("gradient/jacobian/hessian is too close to 0")
        if stopping_criteria(step, rtol=tol):
            warnings.warn(
                f"no further step improvement possible - terminating search "
                f"returning the last step", RuntimeWarning)
            return x 
        if stopping_criteria(f_cur, atol=tol):
            print("root found")
            return x
        
        x = x_next
        f_cur = func(x) 
        g_cur = grad(x)
        step = newton_iteration(f_cur, g_cur)
        x_next = x + step
        funcevals += 1
        gradevals += 1
    warnings.warn(
        f"no solution found in {max_iter} iterations "
        f"returning the last step found", RuntimeWarning)
    return x  

def newton(func, x, grad=None, hess=None, **kwargs):
    if grad is None:
        grad = gradient(func, x, grad, **kwargs)
    hx = hessian(func, x, hess, grad, **kwargs)   
    return hx

def newton_iteration(fx, gx, inv=False):
    '''
    Find better approximation for root of f(x) = 0 than current estimate 
    '''
    if inv:
        if np.ndim(gx) > 1:
            return - gx @ fx 
        else:
            return -gx * fx

    if np.ndim(gx) > 1:
        return np.linalg.solve(gx, -fx)
    
    inv_gx = inverse(gx)
    return - inv_gx * fx

def stopping_criteria(fx, term=0.0, rtol=1e-5, atol=1e-7):
    if np.ndim(fx) == 0:
        value = fx
    else:
        value = l2_norm(fx)
    return np.isclose(value, term, rtol=rtol, atol=atol)


NEWTON_HESSIAN_UPDATES = {
    'newton': newton
}