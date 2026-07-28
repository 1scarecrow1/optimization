import numpy as np
import scipy 


def regularise(hess, adj=None, eps=None):
    d, Q = np.linalg.eigh(hess)

    if eps is not None:
        e = np.sqrt(eps)
    else:
        e = np.finfo(np.float32).eps

    if adj is not None:
        m = adj 
    else: 
        m = np.where(d < e, e - d, 0)
        # or m = np.where(d < 0, e, d)

    modified_hess = Q @ np.diag(d+m) @ Q.T
    return modified_hess

def modified_cholesky(A):
    L, D, perm = scipy.linalg.ldl(A)
    C = np.zeros_like(L)
    n = L.shape[0]
    C[np.diag_indices(n)] = A[np.diag_indices(n)] - np.diag(D) @ L 

    return C

