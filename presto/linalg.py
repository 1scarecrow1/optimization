import numpy as np
from scipy.sparse import csc_array, csc_matrix, diags, issparse, identity
from scipy.sparse.linalg import spilu, SuperLU, spsolve_triangular
import scipy.sparse.linalg as sla
from scipy.linalg import cho_factor, lu_factor, cho_solve, lu_solve
from scipy.linalg.lapack import dlartg

def ssor(A, omega=1.2):
    '''
    Symmetric successive overrelaxation of form A = L + D + L.T for symmetric PD A to solve Ax=b
    '''
    A = np.array(A)
    d = np.diag(A)
    L = np.tril(A, -1)
    D = np.diag(d)
    D_inv = np.diag(1.0 / d)
    b = D + omega*L
    return b @ D_inv @ b.T / (omega * (2-omega))


def incomplete_LU(A, explicit=False, sparse_array=True):
    '''
    Incomplete LU factorisation of square matrix A
    '''
    A = csc_array(A, dtype=float)
    lu = spilu(A)
    if explicit:
        if not sparse_array:
            return lu.L.toarray(), lu.U.toarray()
        return lu.L, lu.U
    return lu

def incomplete_cholesky(A, explicit=False, sparse_array=True, lower=True):
    '''
    Incomplete cholesky factorisation of symmetric PD A
    Uses spilu with no permutation and pivoting 
    spilu returns LU = LDL.T, where U = DL.T -> lower triangular factor is L*D^1/2
    '''
    A = csc_array(A, dtype=float)
    lu = spilu(A, permc_spec='NATURAL', diag_pivot_thresh=0.0) 
    if explicit:
        D = lu.U.diagonal()
        mat = lu.L @ diags(np.sqrt(D))
        if not lower:
            mat = mat.T 
        if not sparse_array:
            return mat.toarray()
        return mat
    return lu
    
def inexact_modified_cholesky(A):
    '''
    T is diag(norms of columns of A). B = T^-1/2 * A * T^-1/2
    Perform Cholesky factorisation on B + alpha*I by incrementally increasing alpha and making 
    diagonal elements more positive
    '''
    t = np.linalg.norm(A, axis=0)
    t_sqrt = np.sqrt(t)
    T_sqrt_inv = np.diag(1/t_sqrt)
    B = T_sqrt_inv @ A @ T_sqrt_inv
    beta = l2_norm(B)

    if np.min(np.diag(B)) > 0:
        alpha = 0.0
    else:
        alpha = beta/2

    I = np.identity(A.shape[0])
    while True:
        try:
            L = np.linalg.cholesky(B + alpha*I)   
            return t_sqrt * L
        except np.linalg.LinAlgError:
            alpha = max(2*alpha, beta/2)

def incomplete_cholesky_shifted(A, alpha0=1e-8, grow=2.0):
    A = csc_array(A, dtype=float)
    diag = A.diagonal()
    scale = np.abs(diag).max()
    alpha = 0.0
    while True:
        try:
            lu = spilu(A + alpha * identity(A.shape[0]), permc_spec='NATURAL', diag_pivot_thresh=0.0)
            if np.all(lu.U.diagonal() > 0):
                return lu
        except RuntimeError:
            pass
        alpha = max(alpha * grow, alpha0 * scale)   


def regularised_cholesky(A, eps=1e-3, factor=10):
    d = np.diag(A)
    I = np.identity(len(d))
    min_d = np.min(d)
    t = 0.0 if min_d > 0 else -min_d + eps
    while True:
        try:
            return np.linalg.cholesky(A + t * I)
        except np.linalg.LinAlgError:
            t = max(factor*t, eps)

def factorised_solver(A, lower=True):
    '''
    Returns (lower-triangular) factor (Cholesky/LU) and prefactorised solver y()
    '''
    if isinstance(A, np.ndarray):
        try:
            cho_fac = cho_factor(A, lower=lower)
            y = lambda x: cho_solve(cho_fac, x)
            # if not lower:
            #     U, lower = cho_factor(A, lower=False)
            #     y = lambda r: cho_solve((U, lower), r)
            # else:
            #     L, lower = cho_factor(A, lower=True)
            #     y = lambda r: cho_solve((L, lower), r)
            M = cho_fac[0]
        except np.linalg.LinAlgError:
            lu, piv = lu_factor(A)
            y = lambda x: lu_solve((lu, piv), x)
            if lower:
                M = np.tril(lu, k=-1) + np.eye(A.shape[0]) 
            else:
                M = np.triu(lu)
        #M = A
    elif isinstance(A, SuperLU): 
        if not lower:
            M = A.U.toarray()
        else:
            M = A.L.toarray()
        y = A.solve
    elif issparse(A): 
        # incomplete Cholesky factor G, A = G*G.T or A = G.T * G
        # Apply M^{-1} = (G Gᵀ)^{-1} via two sparse triangular solves, O(nnz) each
        if not lower:
            Gt = A.tocsr()
            G = Gt.T.tocsr()
            M = Gt
        else:
            G = A.tocsr()
            Gt = G.T.tocsr()
            M = G
        def y(r):
            w = spsolve_triangular(G, r, lower=True)
            return spsolve_triangular(Gt, w, lower=False)
    else:
        raise TypeError('invalid preconditioning method')
    return y, M

def givens_rotations(R_upd, tol=0.0):
    """
    Given an existing Q and an augmented/modified R_upd, restore R_upd to
    upper-triangular form using Givens rotations.

        Q_new = block_diag(Q, I) @ Q_g
        R_new = Q_g.T @ R_upd
        Q_new @ R_new == block_diag(Q, I) @ R_upd

    and the original embedded factors are recoverable by:
        Q_base = Q_new @ Q_g.T
        R_aug_original = Q_g @ R_new
    """
    m, n = R_upd.shape
    R_new = R_upd.copy()
    Q_g = np.eye(m) # gather Givens rotations

    for j in range(min(m, n)):
        for i in range(m - 1, j, -1):
            if abs(R_new[i, j]) <= tol:
                continue
            c, s, _ = dlartg(R_new[i - 1, j], R_new[i, j])

            # apply G to rows i-1 and i of R_new.
            row_top = R_new[i - 1, :].copy()
            row_bot = R_new[i, :].copy()

            R_new[i - 1, :] =  c * row_top + s * row_bot
            R_new[i,     :] = -s * row_top + c * row_bot

            # accumulate Q_g so that:
            # R_new = Q_g.T @ R_upd
            # Q_new = Q_base @ Q_g
            # Since this step used R <- G @ R, accumulate Q_g <- Q_g @ G.T.
            col_left = Q_g[:, i - 1].copy()
            col_right = Q_g[:, i].copy()
            Q_g[:, i - 1] =  c * col_left + s * col_right
            Q_g[:, i]     = -s * col_left + c * col_right

    return Q_g, R_new

def l2_norm(x):
    x = np.asarray(x, dtype=float)
    nx = np.ndim(x)
    if nx == 0:
        return np.abs(x)
    elif nx == 1:
        return np.sqrt(x @ x)
    # elif nx == 2: use SVD, do not square condition number
    #     eigs = np.linalg.eigvalsh(x.T @ x)
    #     return np.sqrt(np.max(eigs))
    elif nx==2:
        return np.linalg.norm(x, 2)
    else:
        raise NotImplementedError(f"not defined for {x.ndim}-dimensional object yet")



def ssor_preconditioner(A, omega=1.2):
    """
    Creates an SSOR preconditioner LinearOperator for iterative solvers.
    A should be SPD.
    """
    if not isinstance(A, csc_matrix):
        A = csc_matrix(A)
    
    n = A.shape[0]
    D = A.diagonal()
    L = -A.multiply(A.indices < A.indptr[:-1, np.newaxis])
    U = -A.multiply(A.indices > A.indptr[:-1, np.newaxis])

    def matvec(b):
        x = np.zeros_like(b, dtype=np.float64)
        
        # 1. Forward Sweep: (D + omega*L) * x_half = omega * b
        for i in range(n):
            sum_val = L[i, :i] @ x[:i]
            x[i] = (omega * b[i] - (omega - 1) * D[i] * x[i] - omega * sum_val) / D[i]
            
        # 2. Backward Sweep: (D + omega*U) * x_new = omega * D * x_half + (1-omega)*D*x_half ...
        # (alternatively, the full SSOR matrix inverse is approximated)
        x_half = x.copy()
        for i in range(n - 1, -1, -1):
            sum_val = U[i, i+1:] @ x_half[i+1:]
            x[i] = (omega * b[i] - (omega - 1) * D[i] * x_half[i] - omega * sum_val) / D[i]
        return x

    return sla.LinearOperator(A.shape, matvec=matvec)

def inverse(x):
    x = np.array(x)
    nx = np.ndim(x)
    if nx <= 1:
        return np.where(np.isclose(x, 0.0), np.inf, 1 / x)  
    elif nx == 2:
        return invert_matrix(x)

def invert_matrix(A):
    if is_diagonal(A):
        return np.diag(1.0/np.diag(A))
    if not is_square(A):
        return moore_penrose_inverse(A)
    return np.linalg.solve(A, np.identity(A.shape[0]))

def moore_penrose_inverse(A):
    if np.ndim(A) != 2:
        raise TypeError("argument must be a matrix with shape (m, n)")
    m, n = A.shape
    if m >= n:
        I = np.identity(n)
        A_inv_l = np.linalg.solve(A.T @ A, I) @ A.T
        return A_inv_l
    I = np.identity(m)
    A_inv_r = A.T @ np.linalg.solve(A @ A.T, I)
    return A_inv_r

def condition_number(x):
    if x.ndim != 2:
        raise ValueError(f"{x} must be a matrix.")
    s = np.linalg.svd(x, compute_uv=False)
    max_sv = np.abs(s[0])
    min_sv = np.abs(s[-1])

    return max_sv / min_sv

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
    return d, q, q

def svd(x):
    a = x.T @ x 
    b = x @ x.T
    _, v = np.linalg.eigh(a)
    d, u = np.linalg.eigh(b)

    return np.sqrt(d), u, v


def solve_quadratic(a, b, c):
    '''
    Solve ax^2 + bx + c = 0
    '''
    disc = b*b - 4*a*c
    if disc < 0:
        raise ValueError(f"no real root: discriminant {disc}")
    q = -0.5 * (b + np.copysign(np.sqrt(disc), b))
    r1, r2 = q / a, c / q
    return max(r1, r2), min(r1, r2)
def solve(A, b):
    if not is_square(A) or not is_PSD(A):
        b = A.T @ b
        A = A.T @ A 
    try:
        return solve_cholesky(A, b)
    except np.linalg.LinAlgError:
        pass

    P, L, U = LUdecomposition_with_pivoting(A)
    b = P @ b
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)

    return x

def solve_cholesky(A, b):
    L = np.linalg.cholesky(A)
    y = forward_substitution(L, b)
    return backward_substitution(L.T, y)

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
    if _is_strictly_diagonally_dominant(A):
        return True
    return rank(A) == A.shape[1]

def is_PD(A):
    return is_positive_definite(A)

def is_PSD(A):
    return is_positive_semi_definite(A)

def is_ND(A):
    return is_negative_definite(A)

def is_NSD(A):
    return is_negative_semi_definite(A)

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
    r = _sym_eigs(A)
    return r is not None and np.all(r[0] > r[1])

def is_positive_semi_definite(A):
    r = _sym_eigs(A)
    return r is not None and np.all(r[0] >= -r[1])

def is_negative_definite(A):
    r = _sym_eigs(A)
    return r is not None and np.all(r[0] < -r[1])

def is_negative_semi_definite(A):
    r = _sym_eigs(A)
    return r is not None and np.all(r[0] <= r[1])

def is_indefinite(A):
    r = _sym_eigs(A)
    return r is not None and np.any(r[0] > r[1]) and np.any(r[0] < -r[1])

def _sym_eigs(A):
    A = np.asarray(A, dtype=float)
    if not is_symmetric(A):
        return None
    w = np.linalg.eigvalsh(A)
    return w, np.finfo(np.float64).eps * max(1.0, np.abs(w).max())   # (eigs, tolerance)

def inertia(A):
    m = np.linalg.eigvals(np.asarray(A, dtype=float))
    pos = np.sum(m > 0)
    neg = np.sum(m < 0)
    return pos, neg, len(m) - pos - neg

def eigenvalues(A):
    if is_diagonal(A):
        return np.sort(np.diag(A))[::-1]
    else:
        raise NotImplementedError


def _is_strictly_diagonally_dominant(A):
    return np.all(np.abs(np.diag(A)) > _off_diagonal_sum(A))

def _off_diagonal_sum(A):
    A = np.array(A, dtype=float)
    return np.sum(np.abs(A), axis=1) - np.diag(np.abs(A))

