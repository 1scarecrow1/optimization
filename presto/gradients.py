from collections.abc import Callable 
from functools import partial
import numpy as np

eps = np.finfo(np.float32).eps

def gradient(fun, x, grad=None, **fun_args):
    if grad is None or isinstance(grad, Callable):
        g = gradient_func(fun, grad, **fun_args)
        return g(x)
    return grad

def hessian(fun, x, hess=None, grad=None, **fun_args):
    if hess is not None and not isinstance(hess, Callable):
        return hess 
    if not isinstance(grad, Callable):
        grad = None 
    h = hessian_func(fun, grad, hess, **fun_args)
    return h(x)

def gradient_func(fun, grad=None, **kwargs):
    if grad is not None:
        return grad 
    if isinstance(fun, partial):
        #fun_args, fun_kwargs = fun.args, fun.keywords
        fun_kwargs = fun.keywords
        fun = fun.func
    else:
        fun_kwargs = kwargs
    if hasattr(fun, 'gradient'):
        return partial(fun.gradient, **fun_kwargs)
    return compute_gradient_func(fun, **fun_kwargs)

def hessian_func(fun, grad=None, hess=None, **kwargs):
    if hess is not None:
        return hess
    if isinstance(fun, partial):
        fun_kwargs = fun.keywords
        fun = fun.func
    else:
        fun_kwargs = kwargs
    if hasattr(fun, 'hessian'):
        return partial(fun.hessian, **fun_kwargs)
    if hasattr(grad, 'gradient'):
        return partial(grad.gradient, **fun_kwargs)
    return compute_hessian_func(fun, grad, **fun_kwargs)

def compute_gradient_func(fun, **fun_args):
    '''
    TODO: Gradient and Hessian approximations 
    Ideally, gradient approximation method should be automatically chosen
    '''
    return partial(finite_difference, fun, central=False, **fun_args)

def compute_hessian_func(fun, grad=None, **fun_args):
    if grad is None:
        grad = compute_gradient_func(fun, **fun_args)
    return partial(finite_difference, fun=grad, central=False, **fun_args)

def finite_difference(fun, x, fx=None, central=False, eps=eps, **fun_args):
    if fx is None:
        fx = fun(x, **fun_args)
    f = partial(fun, **fun_args)
    x = np.array(x, dtype=float)
    #eps = np.sqrt(eps) * (1+np.abs(x)) # increase eps and scale by x
    if np.ndim(fx) == 0:
        if central:
            return gradient_central_finite_difference(f, x, eps=eps)
        return gradient_forward_finite_difference(f, x, fx=fx, eps=eps)
    else:
        if central:
            return jacobian_central_finite_difference(f, x, fx=fx, eps=eps) 
        return jacobian_forward_finite_difference(f, x, fx=fx, eps=eps)


def gradient_central_finite_difference(f, x, eps=eps):
    # f: Rn -> R, x: Rn or R^nxm
    n = np.size(x)
    g = np.zeros_like(x)
    xp = x.copy()
    den = 2*eps
    for i in range(n):
        xi = xp.flat[i]
        xp.flat[i] = xi + eps 
        ff = f(xp)
        xp.flat[i] = xi - eps 
        fb = f(xp)
        g.flat[i] = ff - fb
        xp.flat[i] = xi 
    if np.ndim(x) == 0:
        return g[0]/den
    return g/den

def gradient_forward_finite_difference(f, x, fx=None, eps=eps):
    # f: Rn -> R, x: Rn or R^nxm
    if fx is None:
        fx = f(x)
    n = np.size(x)
    g = np.zeros_like(x)
    xp = x.copy()
    for i in range(n):
        xi = xp.flat[i]
        xp.flat[i] = xi + eps 
        g.flat[i] = f(xp) - fx
        xp.flat[i] = xi 
    if np.ndim(x) == 0:
        return g[0]/eps
    return g/eps

def jacobian_forward_finite_difference(f, x, fx=None, eps=eps):
    # f: Rn -> Rm, x: Rn
    if fx is None:
        fx = f(x)
    m, n = np.size(fx), np.size(x)
    J = np.zeros((m, n))
    xp = x.copy()
    for i in range(n):
        xi = xp.flat[i]
        xp.flat[i] = xi + eps
        J[:, i] = f(xp) - fx
        xp.flat[i] = xi 
    return J/eps

def jacobian_central_finite_difference(f, x, fx=None, eps=eps):
    # f: Rn -> Rm, x: Rn
    if fx is None:
        fx = f(x)
    m, n = np.size(fx), np.size(x)
    den = 2*eps
    J = np.zeros((m, n))
    xp = x.copy()
    for i in range(n):
        xi = xp.flat[i]
        xp.flat[i] = xi + eps 
        ff = f(xp)
        xp.flat[i] = xi - eps 
        fb = f(xp)
        J[:, i] = ff - fb
        xp.flat[i] = xi 

    return J/den

def automatic_differentiation(fun):
    pass