from collections import namedtuple
import warnings
import numpy as np
from presto.linalg import *
from presto.gradients import gradient, hessian, gradient_func, hessian_func
from presto.solvers.newton import newton_iteration
from presto.utils import resolve_func, merge_args
import presto.solvers.quasi_newton as qn
import presto.solvers.newton as nw
import presto.solvers.modified_newton as mn

DescentDirectionOutput = namedtuple("DescentDirectionOutput", "p")
QuasiNewtonOutput = namedtuple("QuasiNewtonOutput", "p B")

def gradient_descent(func, x, grad=None, **func_args):  
    p = - gradient(func, x, grad, **func_args)
    return DescentDirectionOutput(p) 

def newton(func, x, grad=None, hess=None, **func_args):
    gx = gradient(func, x, grad, **func_args)
    hx = hessian(func, x, hess, grad, **func_args)   
    p = newton_iteration(gx, hx)
    return DescentDirectionOutput(p)

def quasi_newton(hessian_approx_method, func, x, grad, 
                 x_next=None, g_next=None, B_cur=None, 
                 inv=True, quasi_newton_args=None, **func_args):
    
    g_cur = gradient(func, x, grad, **func_args)
    if x_next is not None and g_next is None:
        g_next = gradient(func, x_next, grad, **func_args)
    quasi_newton_args = quasi_newton_args or {}
    quasi_newton_method = resolve_func(hessian_approx_method, qn.QUASI_NEWTON_HESSIAN_UPDATES, 'quasi newton hessian update')  
    B_next = quasi_newton_method(x, g_cur, x_next, g_next, B_cur, inv, **quasi_newton_args)
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
    mod_args = mod_args or {}
    mod_newton = resolve_func(mod_method, mn.NEWTON_MODIFICATION_METHODS, 'newton modification methods', regularised_cholesky)
    if mod_newton == mn.gauss_newton_approx:
        try:
            hx = mn.gauss_newton_approx(gx)
            p = np.linalg.solve(hx, -gx)
            return DescentDirectionOutput(p)
        except ValueError:
            warnings.warn('Jacobian has insufficient dimension for Gauss Newton - switching to regularised Cholesky')
            mod_newton = regularised_cholesky
            pass
    hx = hessian(func, x, hess, grad, **func_args)   
    L = mod_newton(hx, **mod_args) 
    if isinstance(L, SuperLU):
        p = L.solve(-gx)
        return DescentDirectionOutput(p)
    import scipy
    y = scipy.linalg.solve_triangular(L, -gx, lower=True)
    p = scipy.linalg.solve_triangular(L.T, y, lower=False)
    return DescentDirectionOutput(p)

def hessian_boundedness(B, upper_bound):
    return condition_number(B) <= upper_bound

GRADIENT_DESCENT = [gradient_descent]
NEWTON = [newton, modified_newton]
QUASI_NEWTON = [quasi_newton, bfgs, symmetric_rank_one, broyden]

NEWTON_SOLVERS = {
    'newton': newton,
    'modified_newton': modified_newton,
}

QUASI_NEWTON_SOLVERS = {
    'bfgs': bfgs,
    'sr1': symmetric_rank_one,
    'broyden': broyden
}


DIRECTIONS = {**NEWTON_SOLVERS, **QUASI_NEWTON_SOLVERS, 'gradient descent': gradient_descent}

def resolve_direction_method(f, methods=DIRECTIONS, name='direction'):
    return resolve_func(f, methods, name)
