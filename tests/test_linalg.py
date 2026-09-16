import numpy as np
import pytest
from presto import linalg as L

rng = np.random.default_rng(0)

def spd(n):
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)

def nonsingular(n):
    while True:
        M = rng.standard_normal((n, n))
        if abs(np.linalg.det(M)) > 1e-6:
            return M

def tridiag(n):
    M = np.zeros((n, n))
    M[np.arange(n), np.arange(n)] = rng.uniform(3, 5, n)
    M[np.arange(1, n), np.arange(n-1)] = rng.uniform(1, 2, n-1)
    M[np.arange(n-1), np.arange(1, n)] = rng.uniform(1, 2, n-1)
    return M

lb = np.eye(4); lb[np.arange(1, 4), np.arange(3)] = [2, 3, 4]
ub = np.eye(4); ub[np.arange(3), np.arange(1, 4)] = [2, 3, 4]

def test_l2_norm():
    for _ in range(5):
        v = rng.standard_normal(6)
        assert np.isclose(L.l2_norm(v), np.linalg.norm(v)), "vector 2-norm"
    for n in (2, 3, 5):
        M = nonsingular(n)
        assert np.isclose(L.l2_norm(M), np.linalg.norm(M, 2)), \
            f"matrix spectral norm n={n}: {L.l2_norm(M)} vs {np.linalg.norm(M, 2)}"
    assert np.isclose(L.l2_norm(np.array(-3.0)), 3.0), "scalar"

def test_predicates():
    assert L.is_square(np.eye(3)), "is_square true"
    assert not L.is_square(rng.standard_normal((2, 3))), "is_square false"
    assert L.is_symmetric(spd(4)), "is_symmetric true"
    assert not L.is_symmetric(nonsingular(4)), "is_symmetric false"
    assert L.is_diagonal(np.diag([1.0, 2, 3])), "is_diagonal true"
    assert not L.is_diagonal(np.array([[1.0, 1], [0, 1]])), "is_diagonal false"
    assert L.is_lower_triangular(np.tril(nonsingular(4))), "is_lower_triangular"
    assert L.is_upper_triangular(np.triu(nonsingular(4))), "is_upper_triangular"
    assert L.is_triangular(np.triu(nonsingular(4))), "is_triangular"
    assert L.is_tridiagonal(tridiag(5)), "is_tridiagonal true"
    assert not L.is_tridiagonal(nonsingular(5)), "is_tridiagonal false"
    assert L.is_bidiagonal(lb), "is_bidiagonal lower"
    assert L.is_bidiagonal(ub), "is_bidiagonal upper"
    assert not L.is_bidiagonal(np.triu(nonsingular(4))), "is_bidiagonal false (full tri)"
    assert L.is_positive_definite(spd(4)), "is_positive_definite true"
    assert not L.is_positive_definite(np.diag([1.0, -2, 3])), "is_positive_definite false"
    assert L.is_symmetric_PD(spd(4)), "is_symmetric_PD true"
    assert L.is_orthogonal(np.linalg.qr(nonsingular(4))[0]), "is_orthogonal true"
    assert not L.is_orthogonal(nonsingular(4)), "is_orthogonal false"
    assert L.is_nonsingular(nonsingular(4)), "is_nonsingular true"
    assert not L.is_nonsingular(np.array([[1.0, 2], [2, 4]])), "is_nonsingular false (singular)"

def test_rank():
    assert L.rank(nonsingular(4)) == 4, "rank full"
    assert L.rank(np.array([[1.0, 2, 3], [2, 4, 6], [0, 1, 1]])) == 2, "rank deficient"
    assert L.rank(np.array([[0.0, 1], [0, 2]])) == 1, "rank tall (col skip)"
    assert L.row_rank(nonsingular(4)) == L.column_rank(nonsingular(4)), "row_rank == col_rank"

def test_gaussian_ref_and_rref():
    A = np.array([[2.0, 1, -1], [-3, -1, 2], [-2, 1, 2]])
    A0 = A.copy()
    ref = L.gaussian_ref(A)
    assert np.allclose(A, A0), "gaussian_ref does not mutate input"
    assert np.allclose(np.tril(ref, -1), 0), "gaussian_ref is upper-triangular"
    R = L.rref(A)
    assert np.allclose(R, np.eye(3)), "rref of nonsingular == I"

def test_substitutions():
    for n in (1, 2, 5):
        Lo = np.tril(nonsingular(n))
        b = rng.standard_normal(n)
        x = L.forward_substitution(Lo, b)
        assert np.allclose(Lo @ x, b), \
            f"forward_substitution n={n} resid={np.linalg.norm(Lo @ x - b):.2e}"
        Up = np.triu(nonsingular(n))
        x = L.backward_substitution(Up, b)
        assert np.allclose(Up @ x, b), \
            f"backward_substitution n={n} resid={np.linalg.norm(Up @ x - b):.2e}"
    x = L.forward_substitution(lb, np.arange(1.0, 5))
    assert np.allclose(lb @ x, np.arange(1.0, 5)), "forward_substitution bidiagonal"
    x = L.backward_substitution(ub, np.arange(1.0, 5))
    assert np.allclose(ub @ x, np.arange(1.0, 5)), "backward_substitution bidiagonal"

def test_forward_substitution_rejects_non_lower():
    with pytest.raises(ValueError):
        L.forward_substitution(np.triu(nonsingular(3)), rng.standard_normal(3))

def test_lu_compact():
    for n in (2, 4, 6):
        A = nonsingular(n)
        B = L.LUdecomposition(A)
        Lm = np.tril(B, -1) + np.eye(n)
        Um = np.triu(B)
        assert np.allclose(Lm @ Um, A), \
            f"compact LU reconstructs n={n} resid={np.linalg.norm(Lm @ Um - A):.2e}"

def test_lu_without_pivoting():
    for n in (2, 4, 6):
        A = spd(n)
        Lm, Um = L.LUdecomposition_without_pivoting(A)
        assert np.allclose(Lm @ Um, A), \
            f"general LU=A n={n} resid={np.linalg.norm(Lm @ Um - A):.2e}"
        assert np.allclose(np.triu(Lm, 1), 0), f"L lower-tri n={n}"
        assert np.allclose(np.tril(Um, -1), 0), f"U upper-tri n={n}"
    for n in (3, 5, 7):
        A = tridiag(n)
        Lm, Um = L.LUdecomposition_without_pivoting(A)
        assert np.allclose(Lm @ Um, A), \
            f"tridiagonal LU=A n={n} resid={np.linalg.norm(Lm @ Um - A):.2e}"

def test_lu_with_pivoting():
    for n in (2, 4, 6):
        A = nonsingular(n)
        P, Lm, Um = L.LUdecomposition_with_pivoting(A)
        assert np.allclose(P @ A, Lm @ Um), \
            f"P@A = L@U n={n} resid={np.linalg.norm(P @ A - Lm @ Um):.2e}"
        assert np.allclose(np.diag(Lm), 1) and np.allclose(np.triu(Lm, 1), 0), \
            f"L unit-lower n={n}"
        assert np.allclose(np.tril(Um, -1), 0), f"U upper n={n}"
        assert np.allclose(np.sort(P.sum(0)), 1) and np.allclose(P @ P.T, np.eye(n)), \
            f"P is permutation n={n}"
    A = nonsingular(4); A0 = A.copy()
    L.LUdecomposition_with_pivoting(A)
    assert np.allclose(A, A0), "with_pivoting does not mutate input"

def test_cholesky():
    for n in (2, 4, 6):
        A = spd(n)
        U = L.cholesky(A)
        assert np.allclose(U.T @ U, A), \
            f"U^T U = A n={n} resid={np.linalg.norm(U.T @ U - A):.2e}"
        assert np.allclose(np.tril(U, -1), 0), f"U upper-tri n={n}"
        Lo = L.cholesky(A, lower=True)
        assert np.allclose(Lo @ Lo.T, A), f"lower=True gives L L^T = A n={n}"

def test_cholesky_rejects_non_pd():
    with pytest.raises(ValueError):
        L.cholesky(np.array([[1.0, 2], [2, 1]]))

def test_solve_and_solve_matrix():
    for n in (2, 4, 6):
        A = nonsingular(n); b = rng.standard_normal(n)
        x = L.solve(A, b)
        assert np.allclose(A @ x, b), \
            f"solve (LU path) n={n} resid={np.linalg.norm(A @ x - b):.2e}"
        S = spd(n); bs = rng.standard_normal(n)
        xs = L.solve(S, bs)
        assert np.allclose(S @ xs, bs), \
            f"solve (Cholesky path) n={n} resid={np.linalg.norm(S @ xs - bs):.2e}"
    for n in (2, 4):
        A = nonsingular(n); Bm = rng.standard_normal((n, 3))
        X = L.solve_matrix(A, Bm)
        assert np.allclose(A @ X, Bm), \
            f"solve_matrix A@X=B n={n} resid={np.linalg.norm(A @ X - Bm):.2e}"

def test_invert_matrix():
    for n in (2, 4, 6):
        A = nonsingular(n)
        Ainv = L.invert_matrix(A)
        assert np.allclose(A @ Ainv, np.eye(n)), \
            f"A @ A^-1 = I n={n} resid={np.linalg.norm(A @ Ainv - np.eye(n)):.2e}"
        assert np.allclose(Ainv, np.linalg.inv(A)), f"inverse matches numpy n={n}"

def test_invert_non_square_is_a_pseudo_inverse():
    A = rng.standard_normal((2, 3))
    X = L.invert_matrix(A)
    assert X.shape == (3, 2), "wide pseudo-inverse shape"
    assert np.allclose(A @ X, np.eye(2)), "A @ A+ = I for wide A"
    B = rng.standard_normal((3, 2))
    Y = L.invert_matrix(B)
    assert Y.shape == (2, 3), "tall pseudo-inverse shape"
    assert np.allclose(Y @ B, np.eye(2)), "A+ @ A = I for tall A"

def test_condition_number():
    for n in (2, 4, 6):
        A = nonsingular(n)
        assert np.isclose(L.condition_number(A), np.linalg.cond(A, 2), rtol=1e-4), \
            f"condition_number ~ numpy n={n}: {L.condition_number(A)} vs {np.linalg.cond(A, 2)}"

def test_spectral_decomposition():
    for n in (3, 5):
        A = spd(n)
        d, q, q2 = L.spectral_decomposition(A)
        assert np.allclose(q @ np.diag(d) @ q.T, A), \
            f"spectral reconstructs n={n} resid={np.linalg.norm(q @ np.diag(d) @ q.T - A):.2e}"

def test_svd_singular_values():
    for n in (3, 4):
        A = nonsingular(n)
        s, u, v = L.svd(A)
        assert np.allclose(np.sort(s), np.sort(np.linalg.svd(A, compute_uv=False))), \
            f"svd singular values match (sorted) n={n}"
