from collections import namedtuple
import numpy as np
from presto.linalg import *
from presto.utils import resolve_func, merge_args

#TODO: resolve grad/hess 

DescentDirectionOutput = namedtuple("DescentDirectionOutput", "p")
QuasiNewtonOutput = namedtuple("QuasiNewtonOutput", "p B")

def gradient_descent(func, x, grad=None, *args, **kwargs):  
    if grad is not None:
        p = - grad
    else:
        p = - func.gradient(x, *args, **kwargs) 
    return DescentDirectionOutput(p) 

def newton(func, x, grad=None, hess=None, *args, **kwargs):
    if grad is not None:
        g = grad 
    else:
        g = func.gradient(x, *args, **kwargs)
    if hess is not None:
        H = hess
    else:
        H = func.hessian(x, *args, **kwargs)
    p = - invert_matrix(H) @ g
    return DescentDirectionOutput(p)

def quasi_newton(func, x_cur, g_cur, x_next=None, g_next=None, B_cur=None, hessian_update_method=None, update_inv=True):
    if x_next is None or g_next is None:
        return QuasiNewtonOutput(-g_cur, np.identity(len(x_cur)))

    if update_inv:
        B_inv = update_inv_hessian(x_cur, g_cur, x_next, g_next, B_cur)
        p = - B_inv @ g_next
        return QuasiNewtonOutput(p, B_inv)

    hessian_update = resolve_hessian_update_method(hessian_update_method) 
    B = hessian_update(x_cur, g_cur, x_next, g_next, B_cur)
    B_inv = invert_matrix(B)

    p = - B_inv @ g_next
    return QuasiNewtonOutput(p, B)

def bfgs(func, x_cur, g_cur, x_next=None, g_next=None, B_cur=None, update_inv=True):
    return quasi_newton(func, x_cur, g_cur, x_next, g_next, B_cur, hessian_update_method='bfgs', update_inv=update_inv)

def bfgs_update(x_cur, g_cur, x_next=None, g_next=None, B_cur=None):
    n = len(x_cur)
    if x_next is None:
        return np.identity(n)

    s = x_next - x_cur 
    y = g_next - g_cur 
    c = y.T @ s
    if c <= 0: # check
        raise ValueError("Secant equation only accepts positive Hessian approximations")
    a = B_cur @ s
    B_next = B_cur - np.outer(a, a.T) / (a.T @ s) + np.outer(y, y.T) / c
    return B_next

def symmetric_rank_one(x_cur, g_cur, x_next=None, g_next=None, B_cur=None):
    n = len(x_cur)
    if x_next is None or g_next is None or B_cur is None:
        return np.identity(n)
    
    s = x_next - x_cur 
    y = g_next - g_cur 
    a = y - B_cur @ s
    c = a.T @ s
    if c <= 0:
        raise ValueError("Secant equation only accepts positive Hessian approximations")
    B_next = B_cur +  np.outer(a, a.T) / c
    return B_next

def update_inv_hessian(x_cur, g_cur, x_next=None, g_next=None, inv_B_cur=None):
    if x_next is None:
        return np.identity(len(x_cur))  
    
    s = x_next - x_cur 
    y = g_next - g_cur 
    c = y.T @ s
    if c <= 0: # check
        raise ValueError("Secant equation only accepts positive Hessian approximations")
    rho = 1 / c
    V_cur = inv_B_cur  
    I = np.identity(len(s))
    a = I - rho * np.outer(s, y)
    V_next = a @ V_cur @ a +rho * np.outer(s, s)
    return V_next

def modified_newton(func, x, grad=None, hess=None, modified_hess=None, mod_method=None, eps=None, *args, **kwargs):
    '''
    Mod or regularisation method
    '''
    if grad is not None:
        g = grad 
    else:
        g = func.gradient(x, *args, **kwargs)
    if hess is not None:
        H = hess
    else:
        H = func.hessian(x, *args, **kwargs)   

    if modified_hess is not None:
        B = modified_hess
    elif mod_method is not None:
        B = mod_method(H, **kwargs) 
    else:
        B = regularise(H, eps=eps)

    p = - invert_matrix(B) @ g
    return DescentDirectionOutput(p)

def regularise(hess, adj=None, eps=None):
    d, Q = np.linalg.eigh(hess)

    if eps is not None:
        e = np.sqrt(eps)
    else:
        e = np.finfo(np.float32).eps

    if adj is not None:
        m = adj 
    else: 
        m = np.where(d < e, e - d, 0)
        # or m = np.where(d < 0, e, d)

    modified_hess = Q @ np.diag(d+m) @ Q.T
    return modified_hess

def regularised_cholesky(A, b=1e-3):
    d = np.diag(A)
    I = np.identity(len(d))
    min_d = np.min(d)

    if min_d > 0:
        t = 0 
    else:
        t = - min_d + b 
    M = A + t * I

    while not is_positive_definite(M):
        min_d = np.min(M)
        if min_d > 0:
            t = 0 
        else:
            t = - min_d + b 
        t = np.max(10*t, b)

        M = A + t * I 

    L = cholesky(M, lower=True)
    return L @ L.T

def modified_cholesky(A):
    L, D = cholesky(A, diag=True)
    C = np.zeros_like(L)
    n = L.shape[0]
    C[np.diag_indices(n)] = A[np.diag_indices(n)] - np.diag(D) @ L 

    return 


def conjugate_gradient(func, x, p_prev, beta, grad=None, *args, **kwargs):
    if grad is not None:
        g = grad 
    else:
        g = func.gradient(x, *args, **kwargs)
    p = - g + beta * p_prev 
    return DescentDirectionOutput(p)

def hessian_boundedness(B, upper_bound):
    return condition_number(B) <= upper_bound

GRADIENT_DESCENT = [gradient_descent]
NEWTON = [newton, modified_newton]
QUASI_NEWTON = [quasi_newton]
HESSIAN_UPDATE = [bfgs, symmetric_rank_one, update_inv_hessian]
INV_HESSIAN_UPDATE = [update_inv_hessian]
CONJUGATE_GRADIENT = [conjugate_gradient]

DIRECTIONS = {
    'gradient_descent': gradient_descent,
    'newton': newton,
    'modified_newton': modified_newton,
    'quasi_newton': quasi_newton,
    'conjugate_gradient': conjugate_gradient
}

SECOND_ORDER_DIRECTIONS = [
    quasi_newton
] 

NEWTON_METHODS = {
    'newton': newton,
    'modified_newton': modified_newton,
    'modified_cholesky': modified_cholesky
}

QUASI_NEWTON_METHODS = {
    'bfgs': bfgs,
}

HESSIAN_UPDATE_METHODS = {
    'bfgs': bfgs_update,
    'symmetric_rank_one': symmetric_rank_one,
    'symmetric_rank_1': symmetric_rank_one,
    'sr1': symmetric_rank_one,
    'inverse_hessian': update_inv_hessian, 
    'inv_hessian': update_inv_hessian,
    'inverse': update_inv_hessian   
}

def resolve_hessian_update_method(f, methods=HESSIAN_UPDATE_METHODS, name='hessian_update'):
    return resolve_func(f, methods, name)