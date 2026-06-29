from presto.linalg import *

def gradient_descent(func, x, grad=None, *args, **kwargs):

    if grad is not None:
        p = - grad
    else:
        p = - func.gradient(x, *args, **kwargs) 
    # if not fun.gradient -> general gradient approximation, same for Hessian
    return p 

def newton(func, x, grad=None, hessian=None, *args, **kwargs):
    if grad is not None:
        g = grad 
    else:
        g = func.gradient(x, *args, **kwargs)
    if hessian is not None:
        H = hessian 
    else:
        H = func.hessian(x, *args, **kwargs)
    return - invert_matrix(H) @ g


def quasi_newton(func, x, *args, **kwargs):
    g = func.gradient(x, *args, **kwargs)
    B = bfgs(func, x)
    p = - B @ g
    return p

def bfgs(func, x):
    pass