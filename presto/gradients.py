from collections.abc import Callable 
from functools import partial

def gradient_func(fun, grad=None, *args, **kwargs):
    if grad is not None:
        return grad 
    if isinstance(fun, partial):
        fun_args, fun_kwargs = fun.args, fun.keywords
        fun = fun.func
    else:
        fun_args, fun_kwargs = args, kwargs
    if hasattr(fun, 'gradient'):
        return partial(fun.gradient, *fun_args, **fun_kwargs)
    return compute_gradient_func(fun, *fun_args, **fun_kwargs)

def hessian_func(fun, hess=None, *args, **kwargs):
    if hess is not None:
        return hess
    if isinstance(fun, partial):
        fun_args, fun_kwargs = fun.args, fun.keywords
        fun = fun.func
    else:
        fun_args, fun_kwargs = args, kwargs
    if hasattr(fun, 'hessian'):
        return partial(fun.hessian, *fun_args, **fun_kwargs)
    return compute_hessian_func(fun, *fun_args, **fun_kwargs)

def compute_gradient_func(func, *args, **kwargs):
    '''
    TODO: Gradient and Hessian approximations 
    '''
    raise NotImplementedError

def compute_hessian_func(func, *args, **kwargs):
    raise NotImplementedError