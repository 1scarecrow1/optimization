from collections import namedtuple
import numpy as np
from presto.linalg import *
from presto.gradients import gradient, hessian, gradient_func, hessian_func
from presto.solvers.newton import newton_iteration
from presto.utils import resolve_func, merge_args
import presto.solvers.quasi_newton as qn
import presto.solvers.newton as nu
from presto.solvers.modified_newton import regularise

#TODO: resolve grad/hess 

DescentDirectionOutput = namedtuple("DescentDirectionOutput", "p")
QuasiNewtonOutput = namedtuple("QuasiNewtonOutput", "p B")

def gradient_descent(func, x, grad=None, **kwargs):  
    p = - gradient(func, x, grad, **kwargs)
    return DescentDirectionOutput(p) 

def newton(func, x, grad=None, hess=None, **kwargs):
    gx = gradient(func, x, grad, **kwargs)
    hx = hessian(func, x, hess, grad, **kwargs)   
    p = newton_iteration(gx, hx)
    return DescentDirectionOutput(p)

def quasi_newton(hessian_approx_method, func, x, grad, 
                 x_next=None, g_next=None, B_cur=None, 
                 inv=True, quasi_newton_args=None, **func_args):
    
    g_cur = gradient(func, x, grad, **func_args)
    if x_next is not None and g_next is None:
        g_next = gradient(func, x_next, grad, **func_args)
    quasi_newton_args = quasi_newton_args or {}
    B_next = hessian_approx_method(x, g_cur, x_next, g_next, B_cur, inv, **quasi_newton_args)
    if g_next is None:
        g_next = g_cur
    p = newton_iteration(g_next, B_next, inv=inv) 
    return QuasiNewtonOutput(p, B_next)

def bfgs(func, x, grad, x_next=None, g_next=None, B_cur=None, inv=True, quasi_newton_args=None, **func_args):
    return quasi_newton(qn.bfgs, func, x, grad, x_next, g_next, B_cur, inv, 
                quasi_newton_args, **func_args)

def symmetric_rank_one(func, x, grad, x_next=None, g_next=None, B_cur=None, inv=True, quasi_newton_args=None, **func_args):
    return quasi_newton(qn.symmetric_rank_one, func, x, grad, x_next, g_next, B_cur, inv, 
                quasi_newton_args, **func_args)

def broyden(func, x, grad, x_next=None, g_next=None, B_cur=None, inv=True, quasi_newton_args=None, **func_args):
    return quasi_newton(qn.broyden, func, x, grad, x_next, g_next, B_cur, inv, 
                quasi_newton_args, **func_args)

def modified_newton(func, x, grad=None, hess=None, mod_method=None, mod_args=None, **func_args):
    gx = gradient(func, x, grad, **func_args)
    hx = hessian(func, x, hess, grad, **func_args)   
    mod_args = mod_args or {}
    mod_method = mod_method or regularised_cholesky
    L = mod_method(hx, **mod_args) 
    y = scipy.linalg.solve_triangular(L, -gx, lower=True)
    p = scipy.linalg.solve_triangular(L.T, y, lower=False)
    return DescentDirectionOutput(p)

# def conjugate_gradient(func, x, p_prev, beta, grad=None, *args, **kwargs):
#     if grad is not None:
#         g = grad 
#     else:
#         g = func.gradient(x, *args, **kwargs)
#     p = - g + beta * p_prev 
#     return DescentDirectionOutput(p)

def hessian_boundedness(B, upper_bound):
    return condition_number(B) <= upper_bound

GRADIENT_DESCENT = [gradient_descent]
NEWTON = [newton, modified_newton]
QUASI_NEWTON = [quasi_newton, bfgs, symmetric_rank_one, broyden]
#CONJUGATE_GRADIENT = [conjugate_gradient]

NEWTON_HESSIAN_UPDATES = [nu.newton]
MODIFIED_NEWTON_HESSIAN_UPDATES = [regularised_cholesky, inexact_modified_cholesky]
QUASI_NEWTON_HESSIAN_UPDATES = [qn.broyden, qn.bfgs, qn.symmetric_rank_one]

NEWTON_METHODS = {
    'newton': newton,
    'modified_newton': modified_newton,
}

QUASI_NEWTON_METHODS = {
    'bfgs': bfgs,
    'sr1': symmetric_rank_one,
    'broyden': broyden
}

NEWTON_MODIFICATION_METHODS = {
    'regularised_cholesky': regularised_cholesky,
    'inexact_modified_cholesky': inexact_modified_cholesky,
    'incomplete_cholesky': incomplete_cholesky,
    'incomplete_LU': incomplete_LU
}

DIRECTIONS = {**NEWTON_METHODS, **QUASI_NEWTON_METHODS, 'gradient_descent': gradient_descent}

def resolve_direction_method(f, methods=DIRECTIONS, name='direction'):
    return resolve_func(f, methods, name)

def resolve_hessian_approx_method(f, methods=QUASI_NEWTON_METHODS, name='hessian_qpprox'):
    return resolve_func(f, methods, name)