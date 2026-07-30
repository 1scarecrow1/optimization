import numpy as np
from presto.linalg import l2_norm
from presto.utils import resolve_func

def quadratic_loss(func, x, fx=None, c=0.5, t=0.0):
    if fx is None:
        fx = func(x)
    tf = t - fx
    if np.ndim(fx) < 1:
        return c*(tf**2) 
    return c*(tf.T @ tf)

def absolute_loss(func, x, fx=None, c=1.0, t=0.0):
    if fx is None:
        fx = func(x)
    tf = t - fx
    if np.ndim(fx) < 1:
        return c*abs(tf) 
    return c*np.linalg.norm(tf, 1)

def huber_loss(func, x, fx=None, delta=0.1, w=0.5):
    '''
    L(y, f(x)) = 0.5 * (y - f(x))**2 if |y - f(x)| <= delta 
                else delta *(|y - f(x)| - 0.5*delta)
    Here func is assumed to be the residual function y - f(x)
    '''
    if fx is None:
        fx = func(x)
    abs_fx = np.abs(fx)
    return np.where(abs_fx <= delta, 
             quadratic_loss(func, x, fx, c=w),
             delta*(absolute_loss(func, x, fx, c=1.0) - (1-w)*delta))

def zero_one_loss(y_act, y_pred):
    return np.where(y_act == y_pred, 0, 1)


LOSS_FUNCTIONS = {
    'squared': quadratic_loss,
    'quadratic': quadratic_loss,
    'sq': quadratic_loss,
    'absolute': absolute_loss,
    'abs': absolute_loss,
    'huber': huber_loss,
}

CATEGORICAL_LOSS_FUNCTIONS = {
    'zero one': zero_one_loss,
    '0-1': zero_one_loss
}

    
