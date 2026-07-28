import numpy as np
import scipy
from scipy.linalg import cho_factor, lu_factor, cho_solve, lu_solve
import presto.conjugate_gradient as cg

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
    return scipy.linalg.solve(R, y, assume_a='upper triangular')

def svd_solve(A, b):
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    y = (1/np.diag(S)) * U.T @ b 
    return np.linalg.solve(Vt, y)

def cg_solve(A, b, x0, preconditioning=True, preconditioner=None, disp_full=False, conv_tol=1e-5, max_iter=100):
    res = cg.optimize.cg_solve(A, b, x0, preconditioning=preconditioning, preconditioner=preconditioner, conv_tol=conv_tol, max_iter=max_iter)
    if disp_full:
        return res 
    return res.x
