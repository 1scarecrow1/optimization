import numpy as np
import scipy 
from presto.linalg import incomplete_LU, incomplete_cholesky, incomplete_cholesky_shifted, regularised_cholesky, inexact_modified_cholesky


def regularise(hess, adj=None, eps=np.finfo(np.float32).eps):
    d, Q = np.linalg.eigh(hess)
    adj = 0.0 if adj is None else adj 
    e = adj + eps
    m = np.where(d < e, e - d, 0)

    while True:
        try:
            modified_hess = Q @ np.diag(d+m) @ Q.T
            return np.linalg.cholesky(modified_hess)
        except np.linalg.LinAlgError:
            e = 2*e 
            m = np.where(d < e, e - d, 0)

def modified_cholesky(A):
    L, D, perm = scipy.linalg.ldl(A)
    C = np.zeros_like(L)
    n = L.shape[0]
    C[np.diag_indices(n)] = A[np.diag_indices(n)] - np.diag(D) @ L 

    return C

def gauss_newton_approx(jac):
    if np.ndim(jac) < 1:
        raise ValueError(f'Jacobian must be atleast 2D')
    return jac.T @ jac

NEWTON_MODIFICATION_METHODS = {
    'regularise': regularise,
    'regularised_cholesky': regularised_cholesky,
    'regularised cholesky': regularised_cholesky,
    'inexact_modified_cholesky': inexact_modified_cholesky,
    'inexact modified cholesky': inexact_modified_cholesky,
    'incomplete_cholesky': incomplete_cholesky,
    'incomplete cholesky': incomplete_cholesky,
    'incomplete_LU': incomplete_LU,
    'incomplete LU': incomplete_LU,
    'incomplete_cholesky_shifted': incomplete_cholesky_shifted,
    'incomplete cholesky shifted': incomplete_cholesky_shifted,
    'gauss_newton': gauss_newton_approx,
    'gauss newton': gauss_newton_approx

}