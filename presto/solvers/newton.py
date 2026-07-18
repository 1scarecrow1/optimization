import numpy as np
from presto.linalg import l2_norm, inverse

machine_eps = np.finfo(np.float32).eps

def minimize(grad, x, hess, max_iter=100, tol=1e-4, eps=machine_eps):
    '''
    Solve f'(x) = 0
    TODO: Implement for vector and matrix valued func
    '''
    x = roots(grad, x, hess, max_iter=max_iter, tol=tol, eps=eps)
    return x  

def roots(func, x, grad, max_iter=100, tol=1e-4, eps=machine_eps):
    '''
    Solve f(x) = 0
    TODO: Implement for vector and matrix valued func
    '''
    x = np.array(x)
    f_cur = func(x)
    g_cur = grad(x)
    step = newton_iteration(f_cur, x, g_cur)
    x_next = x + step

    for _ in range(max_iter):
        if func_stopping_criteria(g_cur, atol=eps):
            raise ZeroDivisionError("gradient/jacobian/hessian is too close to 0")
        if func_stopping_criteria(step, rtol=tol):
            print(f"no further step improvement possible")
            return x_next  
        if func_stopping_criteria(f_cur, atol=tol):
            print("root found")
            return x_next  
        
        x = x_next
        f_cur = func(x) 
        g_cur = grad(x)
        step = newton_iteration(f_cur, x, g_cur)
        x_next = x + step

    print("no solution found")
    return x  

def newton_iteration(fx, gx, inv=False):
    '''
    Find better approximation for root of f(x) = 0 than current estimate 
    '''
    if inv:
        return - gx @ fx 

    if np.ndim(gx) > 1:
        return np.linalg.solve(gx, -fx)
    
    inv_gx = inverse(gx)
    return - inv_gx @ fx

def func_stopping_criteria(fx, term=0.0, rtol=1e-5, atol=1e-7):
    if np.ndim(fx) == 0:
        value = fx
    else:
        value = l2_norm(fx)
    return np.isclose(value, term, rtol=rtol, atol=atol)


