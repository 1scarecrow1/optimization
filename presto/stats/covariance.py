import numpy as np
from presto.linalg import * 

import numpy as np


def ewma_covariance_matrix(returns: list, lam: float) -> list:
    """Compute EWMA covariance matrix from an n_assets x T return matrix.

    Args:
        returns: n_assets x T matrix as list of lists (each row = one asset).
        lam: EWMA decay factor (e.g. 0.94 for daily, 0.97 for weekly).

    Returns:
        n x n covariance matrix as list of lists, each element rounded to 6dp.
    """
    ret = np.array(returns)
    n, t = ret.shape 

    if t == 0:
        return np.zeros((n, n)).tolist()

    demeaned = ret - np.mean(ret, axis=1, keepdims=True)
    weights = np.empty(t, dtype=float)
    weights[0] = lam ** (t - 1)

    if t > 1:
        powers = np.arange(t - 2, -1, -1)
        weights[1:] = (1 - lam) * lam**powers

    covariance = (demeaned * weights) @ demeaned.T

    return np.round(covariance, 6).tolist()

def ewma_covariance_matrix(returns: list, lam: float) -> list:
    ret = np.asarray(returns, dtype=float)
    n, t = ret.shape

    if t == 0:
        return np.zeros((n, n)).tolist()

    ret = ret - ret.mean(axis=1, keepdims=True)

    weights = (1 - lam) * lam ** np.arange(t - 1, -1, -1)
    weights[0] = lam ** (t - 1)

    cov = (ret * weights) @ ret.T

    return np.round(cov, 6).tolist()

import numpy as np
from scipy.signal import lfilter

def solution(data, alpha):
    # TODO: Implement EMA using scipy.signal.lfilter
    # Ensure initialization matches y[0] = x[0]
    if not data:
        return np.empty()

    x = np.array(data)
    t = len(x)
    a = [1, alpha - 1]
    b = [alpha]


    # lfilter gives y[0] = alpha*x[0] + zi[0].
    # Choose zi so y[0] = x[0].
    zi = [(1 - alpha) * x[0]]

    y, _ = lfilter(b, a, x, zi=zi)

    return y

def covariance(x, y, conditional_on=False, correction=False):
    if x.ndim == 1 and y.ndim == 1:
        return vector_covariance(x, y, conditional_on=conditional_on, correction=correction)
    else:
        raise NotImplementedError
    
def vector_covariance(x, y, conditional_on=False, ddof=1):
    n = len(x)
    s = (x - np.nanmean(x)) @ (y - np.nanmean(y))
    return s / (n-ddof)   

def sample_mean(x):
    x = np.array(x, dtype=float)
    nx = x.ndim
    if nx == 0:
        return x
    elif nx == 1:
        return np.mean(x)
    elif nx == 2:
        return np.mean(x, axis=1)
    else:
        raise NotImplementedError

def sample_covariance(x, *args, ddof=1):
    nx = x.ndim
    y = np.hstack(x, args)
    if y:
        ny = y.ndim
        if nx == ny == 1:
            return vector_covariance(x, y)
        elif nx > 0:
            pass 
    return 


def sample_covariance_matrix(x):
    if x.ndim < 1:
        return 0.0
    elif x.ndim == 1:
        return np.nanvar(x, ddof=1)
    elif x.ndim == 2:
        return covariance_matrix(x)
    else:
        raise NotImplementedError
    
def sample_correlation_matrix(x):
    y = demean(x)
    y_norm = l2_norm(y)
    v = np.diag(1 / y_norm)
    return v @ y.T @ y @ v

def covariance_matrix(x):
    n, k = x.shape
    if n < 2 or k < 2:
        raise ValueError("Need at least 2 observations for each variable")
    y = demean(x)
    return (y.T @ y) / (n - 1)

def demean(x):
    return x - np.nanmean(x)