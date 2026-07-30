import warnings
import numpy as np
import scipy
from scipy.linalg import cho_factor, lu_factor, cho_solve, lu_solve
import presto.conjugate_gradient.optimize as cg
from presto.utils import resolve_func
from presto.linalg import backward_substitution

def cholesky_solve(A, b):
    coef = A.T @ A
    y = A.T @ b
    try:
        cfac = cho_factor(coef)
    except np.linalg.LinAlgError:
        print("Cholesky factorization breakdown - ill conditional matrix")
    return cho_solve(cfac, y)

def qr_solve(A, b):
    Q, R = np.linalg.qr(A)
    y = Q.T @ b 
    return backward_substitution(R, y)

def svd_solve(A, b):
    U, s, Vt = np.linalg.svd(A)
    k = s.size 
    if k == U.shape[1]:
        U = (1/s) * U 
    else:
        Vt = (1/s) * Vt  
    return Vt.T @ U.T @ b 

def cg_solve(A, b, x0, preconditioning=True, preconditioner=None, disp_full=False, conv_tol=1e-5, max_iter=100):
    res = cg.cg_solve(A, b, x0, preconditioning=preconditioning, preconditioner=preconditioner, conv_tol=conv_tol, max_iter=max_iter)
    if disp_full:
        return res 
    return res.x

LINEAR_LSTSQ_METHODS = {
    'cholesky': cholesky_solve,
    'qr': qr_solve,
    'svd': svd_solve,
    'cg': cg_solve,
}

NON_ITERATIVE_SOLVERS = [cholesky_solve, qr_solve, svd_solve]

