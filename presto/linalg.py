import numpy as np
from presto.utils import timer

def l2_norm(x):
    nx = x.ndim
    if nx == 0:
        return np.abs(x)
    elif nx == 1:
        return np.sqrt(x @ x)
    elif nx == 2:
        eigs = np.linalg.eigvalsh(x.T @ x)
        return np.sqrt(np.max(eigs))
    else:
        raise NotImplementedError(f"not defined for {x.ndim}-dimensional object yet")
    
def invert_matrix(A):
    if not is_square(A):
        raise ValueError("Matrix must be square")
    if not is_nonsingular(A):
        raise ValueError("Matrix is singular")
    return solve_matrix(A, np.identity(A.shape[0]))

def condition_number(x):
    if x.ndim != 2:
        raise ValueError(f"{x} must be a matrix.")
    
    max_sv = l2_norm(x)
    inv_min_sv = l2_norm(invert_matrix(x))

    return max_sv * inv_min_sv

def eigendecomposition(A):
    if is_symmetric(A):
        return spectral_decomposition(A)
    else:
        return svd(A)
    
def spectral_decomposition(A):
    '''
    TODO: implement eigenvalue/vector
    '''
    d, q = np.linalg.eigh(A)
    # q @ np.diag(d) @ q.T
    return d, q, q

def svd(x):
    '''
    TODO: Fix left and right eigenvectors
    '''
    a = x.T @ x 
    b = x @ x.T
    _, v = np.linalg.eigh(a)
    d, u = np.linalg.eigh(b)
    #u @ np.diag(np.sqrt(d)) @ v.T

    return np.sqrt(d), u, v


def solve(A, b):
    if is_positive_definite(A):
        return solve_cholesky(A, b)
    elif is_square(A) and (not 0 in np.diag(A)):
        P, L, U = LUdecomposition_with_pivoting(A)
        b = P @ b
    else:
        raise NotImplementedError

    y = forward_substitution(L, b)
    x = backward_substitution(U, y)

    return x

def solve_cholesky(A, b):
    if not is_positive_definite(A):
        raise ValueError(f"{A} must be symmetric positive definite")
    U = cholesky(A)
    y = forward_substitution(U.T, b)
    x = backward_substitution(U, y)
    return x

def solve_matrix(A, B):
    n, p = B.shape
    P, L, U = LUdecomposition_with_pivoting(A)
    Y, X = np.zeros_like(B), np.zeros_like(B)

    for i in range(p):
        Y[:, i] = forward_substitution(L, P @ B[:, i])
        X[:, i] = backward_substitution(U, Y[:, i])

    return X

def cholesky(A, lower=False):
    '''
    TODO: LU decomp, LDLT etc
    '''
    A = np.array(A, dtype=float)
    if not is_positive_definite(A):
        raise ValueError(f"Matrix must be symmetric PD")
    n = A.shape[0]
    U = np.zeros_like(A) 
    for i in range(n):
        U[i, i] = np.sqrt(A[i, i])
        U[i, i+1:] = A[i, i+1:] / U[i, i]
        A[i+1:, i+1:] -= U[i, i+1:, None] * U[i, i+1:]

    if lower:
        return U.T
    return U

def LUdecomposition(A):
    A = np.asarray(A, dtype=float)
    n = len(A)
    B = A.copy()
    for k in range(1, n):
        for i in range(k):
            c = B[k, i] / B[i, i]
            B[k, i] = c
            B[k, i+1:] += - B[i, i+1:] * c
    return B

def LUdecomposition_without_pivoting(A):
    '''
    TODO: handle non-square matrices
    '''
    A = np.array(A, dtype=float)
    if not is_nonsingular(A):
        raise ValueError(f"{A} must be non-singular")
    U = np.zeros_like(A)
    L = np.zeros_like(A)
    n = A.shape[0]

    if is_tridiagonal(A):
        L[np.diag_indices(n)] = 1
        U[np.diag_indices(n)] = A[np.diag_indices(n)]
        for i in range(n-1):
            L[i+1, i] = A[i+1, i] / A[i, i]
            U[i, i:i+2] = A[i, i:i+2]
            A[i+1, i+1] -= L[i+1, i]*U[i, i+1]
    else:
        for i in range(n-1):
            U[i, i:] = A[i, i:]
            L[i:, i] = A[i:, i] / U[i, i]
            A[i+1:, i+1:] -= L[i+1:, i, None] * U[i, i+1:]
        L[-1, -1] = 1
        
    U[-1, -1] = A[-1, -1]

    return L, U  

def LUdecomposition_with_pivoting(A):
    '''
    TODO: LAPACK/BLAS or JIT the loop with Numba
    '''
    A = np.array(A, dtype=float)
    if not is_square(A):
        raise ValueError(f"{A} must be square")

    n = A.shape[0]
    P, L = np.identity(n), np.identity(n)
    U = np.zeros_like(A)

    for i in range(n-1):
        i_max = np.argmax(np.abs(A[i:, i])) + i
        A[[i, i_max]] = A[[i_max, i]]
        P[[i, i_max]] = P[[i_max, i]]
        L[[i, i_max], :i] = L[[i_max, i], :i]
        L[i:, i] = A[i:, i] / A[i, i]
        U[i, i:] = A[i, i:]
        A[i+1:, i+1:] -= L[i+1:, i, None] * U[i, i+1:]

    L[-1, -1] = 1
    U[-1, -1] = A[-1, -1]
    return P, L, U

def gaussian_ref(A, pivoting=False):
    '''
    TODO: Vectorise further, remove while loop if possible, do with low-level Numba
    '''
    A = np.array(A, dtype=float)
    m, n = A.shape
    h = k = 0
    while h < m and k < n:
        i_max = np.argmax(np.abs(A[h:, k])) + h
        if A[i_max, k] == 0:
            k += 1
            continue
        A[[h, i_max]] = A[[i_max, h]]

        A[h+1:, k] /= A[h, k]
        A[h+1:, k+1:] -= A[h+1:, k, None] * A[h, k+1:]
        A[h+1:, k] = 0
        h += 1
        k += 1
    return A

def gaussian_elimination(A, *, pivoting=True):
    """
    Gaussian elimination, once, in place on a copy.

    Returns (M, piv, pcols, perm, sign)
    """ 

    return 



def rref(A, ref=False):
    if not ref:
        A = gaussian_ref(A)

    A = np.array(A, dtype=float)
    m, _  = A.shape
    for i in range(m-1, -1, -1):
        nz = np.flatnonzero(A[i])
        if len(nz) == 0:
            continue

        k = nz[0]
        A[i] /= A[i, k]
        A[:i] -= A[:i, k, None] * A[i]
        A[:i, k] = 0

    return A

def gaussian_pivots(A):
    A = np.array(A, dtype=float)
    A = gaussian_ref(A)
    m = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        nz = np.flatnonzero(A[i])
        if len(nz) == 0:
            m[i] = 0
        else:
            m[i] = A[i, nz[0]]
    return m       

    
def forward_substitution(A, b):
    A = np.array(A, dtype=float)

    if not is_lower_triangular(A):
        raise ValueError(f"{A} must be lower triangular")
    
    if not is_nonsingular(A):
        raise ValueError(f"{A} must be nonsingular")

    l = np.diag(A)
    if is_diagonal(A):
        return b / l

    n = len(l)
    x = np.zeros(n, dtype=float)
    x[0] = b[0] / l[0]

    if is_bidiagonal(A):
        for i in range(1, n):
            x[i] = (b[i] - A[i, i-1] * x[i-1]) / l[i]
    else:      
        for i in range(1, n):
            x[i] = (b[i] - A[i, :i] @ x[:i]) / l[i]
    return x  


def backward_substitution(A, b):
    A = np.array(A, dtype=float)

    if not is_upper_triangular(A):
        raise ValueError(f"{A} must be upper triangular")
    
    if not is_nonsingular(A):
        raise ValueError(f"{A} must be nonsingular")

    u = np.diag(A)
    if is_diagonal(A):
        return b / u
    
    n = len(u)
    x = np.zeros(n, dtype=float)
    x[n-1] = b[n-1] / u[n-1]

    if is_bidiagonal(A):
        for i in range(n-2, -1, -1):
            x[i] = (b[i] - A[i, i+1] * x[i+1]) / u[i]
    else:
        for i in range(n-2, -1, -1):
            x[i] = (b[i] - A[i, i+1:] @ x[i+1:]) / u[i]
    return x


def rank(A):
    if is_triangular(A):
        return np.sum(~np.isclose(np.diag(A), 0))
    
    ref = gaussian_ref(A)
    return np.sum(~np.all(np.isclose(ref, 0), axis=1))

def column_rank(A):
    return rank(A)

def row_rank(A):
    return rank(A.T)

def is_square(A):
    A = np.array(A, dtype=float)
    return A.shape[0] == A.shape[1]

def is_diagonal(A):
    A = np.array(A, dtype=float)
    i, j = np.nonzero(~np.isclose(A, 0))
    return np.all(i == j)

def is_bidiagonal(A):
    if not (is_lower_triangular(A) or is_upper_triangular(A)):
        return False     
    if is_lower_triangular(A):
        return np.allclose(np.tril(A, k=-2), 0) and not(np.allclose(np.diag(A, k=-1), 0))    
    if is_upper_triangular(A):
        return np.allclose(np.triu(A, k=2), 0) and not(np.allclose(np.diag(A, k=1), 0))    
    return False

def is_tridiagonal(A):
    A = np.array(A, dtype=float)
    nonzero_diags =  not(np.all(np.isclose(np.diag(A, k=0), 0)) 
                         or np.all(np.isclose(np.diag(A, k=-1), 0)) 
                         or np.all(np.isclose(np.diag(A, k=1), 0)))
    zeros_beyond = np.allclose(np.tril(A, k=-2), 0) and np.allclose(np.triu(A, k=2), 0)
    return nonzero_diags and zeros_beyond

def is_lower_triangular(A):
    A = np.array(A, dtype=float)
    return is_diagonal(A) or np.allclose(np.triu(A, k=1), 0)

def is_upper_triangular(A):
    A = np.array(A, dtype=float)
    return is_diagonal(A) or np.allclose(np.tril(A, k=-1), 0)

def is_triangular(A):
    return is_lower_triangular(A) or is_upper_triangular(A)

def is_nonsingular(A):
    if not is_square(A):
        return False
    if is_triangular(A):
        return not (0 in np.diag(A))    
    # if _is_strictly_diagonally_dominant(A):
    #     return True
    return rank(A) == A.shape[1]

def is_PD(A):
    return is_positive_definite(A)

def is_PSD(A):
    return is_positive_semi_definite(A)

def is_ND(A):
    return is_positive_definite(A)

def is_NSD(A):
    return is_positive_semi_definite(A)

def is_symmetric(A):
    return is_square(A) and np.allclose(A, A.T)
    
def is_symmetric_PD(A):
    return is_symmetric(A) and is_positive_definite(A) 

def is_orthogonal(A):
    if not is_square(A):
        return False
    n = A.shape[0]
    I = np.identity(n)
    return np.allclose(A @ A.T, I) and np.allclose(A.T @ A, I) 

def is_orthonally_diagonalisable(A):
    return is_symmetric(A)  

def is_positive_definite(A):
    if not is_orthonally_diagonalisable(A):
        return False
    return np.all(gaussian_pivots(A) > 0)

def is_positive_semi_definite(A):
    if not is_orthonally_diagonalisable(A):
        return False
    return np.all(gaussian_pivots(A) >= 0)

def is_negative_definite(A):
    if not is_orthonally_diagonalisable(A):
        return False
    return np.all(gaussian_pivots(A) < 0)

def is_negative_semi_definite(A):
    if not is_orthonally_diagonalisable(A):
        return False
    return np.all(gaussian_pivots(A) <= 0)

def is_indefinite(A):
    if not is_orthonally_diagonalisable(A):
        return False
    return not (is_positive_semi_definite(A) or is_negative_semi_definite(A))

def inertia(A):
    m = gaussian_pivots(A)
    pos = np.sum(m > 0)
    neg = np.sum(m < 0)
    return pos, neg, len(m) - pos - neg

def eigenvalues(A):
    if is_diagonal(A):
        return np.sort(np.diag(A), kind='heapsort', descending=True)
    else:
        raise NotImplementedError


# use np.diag_indices instead

def _is_strictly_diagonally_dominant(A):
    #return np.diag(A) > _off_diagonal_sum(A)
    abs_A = np.abs(A)
    return np.sum(abs_A, axis=1) < 2*np.diag(abs_A)

def _off_diagonal_sum(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    return np.sum(np.abs(A), axis=1) - np.diag(np.abs(A))

