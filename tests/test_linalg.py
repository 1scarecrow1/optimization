"""
Ad-hoc test harness for presto/linalg.py — run: python test_linalg.py
Compares each implemented function against numpy references.
Uses a fixed RNG seed so failures are reproducible.
svd reconstruction is only sanity-checked on singular values (sign/order is a known TODO).
"""
import numpy as np
from presto import linalg as L

rng = np.random.default_rng(0)
PASS, FAIL = 0, 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")

def check_raises(name, fn, exc=Exception):
    global PASS, FAIL
    try:
        fn()
    except exc:
        PASS += 1
        print(f"  ok   {name} (raised as expected)")
    except Exception as e:
        FAIL += 1
        print(f"  FAIL {name}  wrong exception: {type(e).__name__}: {e}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  no exception raised")

def spd(n):
    """Random symmetric positive-definite matrix."""
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)

def nonsingular(n):
    """Random well-conditioned nonsingular matrix."""
    while True:
        M = rng.standard_normal((n, n))
        if abs(np.linalg.det(M)) > 1e-6:
            return M

def tridiag(n):
    M = np.zeros((n, n))
    M[np.arange(n), np.arange(n)] = rng.uniform(3, 5, n)          # diag
    M[np.arange(1, n), np.arange(n-1)] = rng.uniform(1, 2, n-1)   # sub
    M[np.arange(n-1), np.arange(1, n)] = rng.uniform(1, 2, n-1)   # super
    return M

# ---------------------------------------------------------------- l2_norm
print("l2_norm")
for _ in range(5):
    v = rng.standard_normal(6)
    check("vector 2-norm", np.isclose(L.l2_norm(v), np.linalg.norm(v)))
for n in (2, 3, 5):
    M = nonsingular(n)
    check(f"matrix spectral norm n={n}",
          np.isclose(L.l2_norm(M), np.linalg.norm(M, 2)),
          f"{L.l2_norm(M)} vs {np.linalg.norm(M, 2)}")
check("scalar", np.isclose(L.l2_norm(np.array(-3.0)), 3.0))

# ---------------------------------------------------------------- predicates
print("predicates")
check("is_square true",  L.is_square(np.eye(3)))
check("is_square false", not L.is_square(rng.standard_normal((2, 3))))
check("is_symmetric true",  L.is_symmetric(spd(4)))
check("is_symmetric false", not L.is_symmetric(nonsingular(4)))
check("is_diagonal true",  L.is_diagonal(np.diag([1.0, 2, 3])))
check("is_diagonal false", not L.is_diagonal(np.array([[1.0, 1], [0, 1]])))
check("is_lower_triangular", L.is_lower_triangular(np.tril(nonsingular(4))))
check("is_upper_triangular", L.is_upper_triangular(np.triu(nonsingular(4))))
check("is_triangular", L.is_triangular(np.triu(nonsingular(4))))
check("is_tridiagonal true", L.is_tridiagonal(tridiag(5)))
check("is_tridiagonal false", not L.is_tridiagonal(nonsingular(5)))
# lower / upper bidiagonal
lb = np.eye(4); lb[np.arange(1, 4), np.arange(3)] = [2, 3, 4]
ub = np.eye(4); ub[np.arange(3), np.arange(1, 4)] = [2, 3, 4]
check("is_bidiagonal lower", L.is_bidiagonal(lb))
check("is_bidiagonal upper", L.is_bidiagonal(ub))
check("is_bidiagonal false (full tri)", not L.is_bidiagonal(np.triu(nonsingular(4))))
check("is_positive_definite true", L.is_positive_definite(spd(4)))
check("is_positive_definite false", not L.is_positive_definite(np.diag([1.0, -2, 3])))
check("is_symmetric_PD true", L.is_symmetric_PD(spd(4)))
check("is_orthogonal true", L.is_orthogonal(np.linalg.qr(nonsingular(4))[0]))
check("is_orthogonal false", not L.is_orthogonal(nonsingular(4)))
check("is_nonsingular true", L.is_nonsingular(nonsingular(4)))
check("is_nonsingular false (singular)",
      not L.is_nonsingular(np.array([[1.0, 2], [2, 4]])))

# ---------------------------------------------------------------- rank
print("rank")
check("rank full", L.rank(nonsingular(4)) == 4)
check("rank deficient", L.rank(np.array([[1.0, 2, 3], [2, 4, 6], [0, 1, 1]])) == 2)
check("rank tall (col skip)", L.rank(np.array([[0.0, 1], [0, 2]])) == 1)
check("row_rank == col_rank", L.row_rank(nonsingular(4)) == L.column_rank(nonsingular(4)))

# ---------------------------------------------------------------- gaussian_ref / rref
print("gaussian_ref / rref")
A = np.array([[2.0, 1, -1], [-3, -1, 2], [-2, 1, 2]])
A0 = A.copy()
ref = L.gaussian_ref(A)
check("gaussian_ref does not mutate input", np.allclose(A, A0))
check("gaussian_ref is upper-triangular", np.allclose(np.tril(ref, -1), 0))
R = L.rref(A)
# rref of a nonsingular square matrix is the identity
check("rref of nonsingular == I", np.allclose(R, np.eye(3)))

# ---------------------------------------------------------------- substitutions
print("substitutions")
for n in (1, 2, 5):
    Lo = np.tril(nonsingular(n))
    b = rng.standard_normal(n)
    x = L.forward_substitution(Lo, b)
    check(f"forward_substitution n={n}", np.allclose(Lo @ x, b),
          f"resid={np.linalg.norm(Lo @ x - b):.2e}")
    Up = np.triu(nonsingular(n))
    x = L.backward_substitution(Up, b)
    check(f"backward_substitution n={n}", np.allclose(Up @ x, b),
          f"resid={np.linalg.norm(Up @ x - b):.2e}")
# bidiagonal fast paths
x = L.forward_substitution(lb, np.arange(1.0, 5))
check("forward_substitution bidiagonal", np.allclose(lb @ x, np.arange(1.0, 5)))
x = L.backward_substitution(ub, np.arange(1.0, 5))
check("backward_substitution bidiagonal", np.allclose(ub @ x, np.arange(1.0, 5)))
# error paths
check_raises("forward rejects non-lower",
             lambda: L.forward_substitution(np.triu(nonsingular(3)), rng.standard_normal(3)),
             ValueError)

# ---------------------------------------------------------------- LU (compact Doolittle)
print("LUdecomposition (compact)")
for n in (2, 4, 6):
    A = nonsingular(n)
    B = L.LUdecomposition(A)
    Lm = np.tril(B, -1) + np.eye(n)
    Um = np.triu(B)
    check(f"compact LU reconstructs n={n}", np.allclose(Lm @ Um, A),
          f"resid={np.linalg.norm(Lm @ Um - A):.2e}")

# ---------------------------------------------------------------- LU without pivoting
print("LUdecomposition_without_pivoting")
for n in (2, 4, 6):
    A = spd(n)  # SPD => no pivoting needed, no zero pivots
    Lm, Um = L.LUdecomposition_without_pivoting(A)
    check(f"general LU=A n={n}", np.allclose(Lm @ Um, A),
          f"resid={np.linalg.norm(Lm @ Um - A):.2e}")
    check(f"L lower-tri n={n}", np.allclose(np.triu(Lm, 1), 0))
    check(f"U upper-tri n={n}", np.allclose(np.tril(Um, -1), 0))
# tridiagonal branch
for n in (3, 5, 7):
    A = tridiag(n)
    Lm, Um = L.LUdecomposition_without_pivoting(A)
    check(f"tridiagonal LU=A n={n}", np.allclose(Lm @ Um, A),
          f"resid={np.linalg.norm(Lm @ Um - A):.2e}")

# ---------------------------------------------------------------- LU with pivoting
print("LUdecomposition_with_pivoting")
for n in (2, 4, 6):
    A = nonsingular(n)
    P, Lm, Um = L.LUdecomposition_with_pivoting(A)
    check(f"P@A = L@U n={n}", np.allclose(P @ A, Lm @ Um),
          f"resid={np.linalg.norm(P @ A - Lm @ Um):.2e}")
    check(f"L unit-lower n={n}",
          np.allclose(np.diag(Lm), 1) and np.allclose(np.triu(Lm, 1), 0))
    check(f"U upper n={n}", np.allclose(np.tril(Um, -1), 0))
    check(f"P is permutation n={n}",
          np.allclose(np.sort(P.sum(0)), 1) and np.allclose(P @ P.T, np.eye(n)))
# input must not be mutated
A = nonsingular(4); A0 = A.copy()
L.LUdecomposition_with_pivoting(A)
check("with_pivoting does not mutate input", np.allclose(A, A0))

# ---------------------------------------------------------------- cholesky
print("cholesky")
for n in (2, 4, 6):
    A = spd(n)
    U = L.cholesky(A)
    check(f"U^T U = A n={n}", np.allclose(U.T @ U, A),
          f"resid={np.linalg.norm(U.T @ U - A):.2e}")
    check(f"U upper-tri n={n}", np.allclose(np.tril(U, -1), 0))
    Lo = L.cholesky(A, lower=True)
    check(f"lower=True gives L L^T = A n={n}", np.allclose(Lo @ Lo.T, A))
check_raises("cholesky rejects non-PD",
             lambda: L.cholesky(np.array([[1.0, 2], [2, 1]])), ValueError)

# ---------------------------------------------------------------- solve / solve_matrix
print("solve / solve_matrix")
for n in (2, 4, 6):
    A = nonsingular(n); b = rng.standard_normal(n)
    x = L.solve(A, b)
    check(f"solve (LU path) n={n}", np.allclose(A @ x, b),
          f"resid={np.linalg.norm(A @ x - b):.2e}")
    S = spd(n); bs = rng.standard_normal(n)
    xs = L.solve(S, bs)
    check(f"solve (Cholesky path) n={n}", np.allclose(S @ xs, bs),
          f"resid={np.linalg.norm(S @ xs - bs):.2e}")
for n in (2, 4):
    A = nonsingular(n); Bm = rng.standard_normal((n, 3))
    X = L.solve_matrix(A, Bm)
    check(f"solve_matrix A@X=B n={n}", np.allclose(A @ X, Bm),
          f"resid={np.linalg.norm(A @ X - Bm):.2e}")

# ---------------------------------------------------------------- invert / condition
print("invert_matrix / condition_number")
for n in (2, 4, 6):
    A = nonsingular(n)
    Ainv = L.invert_matrix(A)
    check(f"A @ A^-1 = I n={n}", np.allclose(A @ Ainv, np.eye(n)),
          f"resid={np.linalg.norm(A @ Ainv - np.eye(n)):.2e}")
    check(f"inverse matches numpy n={n}", np.allclose(Ainv, np.linalg.inv(A)))
check_raises("invert rejects non-square",
             lambda: L.invert_matrix(rng.standard_normal((2, 3))), ValueError)
for n in (2, 4, 6):
    A = nonsingular(n)
    check(f"condition_number ~ numpy n={n}",
          np.isclose(L.condition_number(A), np.linalg.cond(A, 2), rtol=1e-4),
          f"{L.condition_number(A)} vs {np.linalg.cond(A, 2)}")

# ---------------------------------------------------------------- spectral / svd
print("spectral_decomposition / svd")
for n in (3, 5):
    A = spd(n)
    d, q, q2 = L.spectral_decomposition(A)
    check(f"spectral reconstructs n={n}", np.allclose(q @ np.diag(d) @ q.T, A),
          f"resid={np.linalg.norm(q @ np.diag(d) @ q.T - A):.2e}")
# svd sign/order is a known TODO -> only check the singular VALUES (set-wise)
for n in (3, 4):
    A = nonsingular(n)
    s, u, v = L.svd(A)
    check(f"svd singular values match (sorted) n={n}",
          np.allclose(np.sort(s), np.sort(np.linalg.svd(A, compute_uv=False))),
          "sign/order of U,V is a documented TODO")

# ---------------------------------------------------------------- summary
print("\n" + "=" * 40)
print(f"PASSED: {PASS}   FAILED: {FAIL}")
if FAIL:
    raise SystemExit(1)
