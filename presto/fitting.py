import numpy as np
from presto.linalg import * 

def linear_interpolation(x, b):
    x = np.sort(x, kind='heapsort')
    n = len(x)
    f = np.zeros_like(x)
    f = (b[None, 1:]*(x[1:] - x[:-1]) + b[None, :-1] * (x[1:] - x[:-1])) / (x[1:] - x[:-1])

    return f

def cubic_spline_interpolation(x, v):
    n = len(x)
    y = np.zeros(4*n)
    b = np.zeros(4*n)
    A = np.zeros(4*n, 4*n)
    A[0, 2] = 2
    A[0, 3] = 6 * x[0]
    A[-1, -2] = 2
    A[-1, -1] = 6 * x[-1]
    for i in range(n):
        b[4*i - 2] = v[i-1]
        b[4*i - 1] = v[i]
        A[4*i - 2, 4*i - 3] = 1
        A[4*i - 2, 4*i - 3] = 1     
    return  
