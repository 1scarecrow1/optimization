from collections.abc import Callable 
from functools import partial

def gradient(fun, x, grad=None, **fun_args):
    if grad is None or isinstance(grad, Callable):
        g = gradient_func(fun, **fun_args)
        return g(x)
    return grad

def hessian(fun, x, hess=None, grad=None, **fun_args):
    if hess is None or isinstance(hess, Callable):
        h = hessian_func(fun, grad, hess, **fun_args)
        return h(x)
    return hess

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
    return compute_hessian_func(fun, grad, **fun_kwargs)

def compute_gradient_func(func, *args, **kwargs):
    '''
    TODO: Gradient and Hessian approximations 
    Ideally, gradient approximation method should be automatically chosen
    '''
    raise NotImplementedError

def compute_hessian_func(func, grad, *args, **kwargs):
    # use gradient info if available
    raise NotImplementedError


def finite_difference(fun, central=False):

    if central:
        return central_finite_difference(fun)

    return forward_finite_difference(fun)

def central_finite_difference(fun):
    pass 

def forward_finite_difference(fun):
    pass

def automatic_differentiation(fun):
    pass