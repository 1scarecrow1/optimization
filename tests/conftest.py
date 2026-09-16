"""
Contains test functions with analytic derivatives, matrix generators covering 
the three cases a trust-region subproblem solver has to handle, and an independent 
reference solver for that subproblem so the fast solvers can be checked against 
the exact answer rather than against each other.
"""
import matplotlib
matplotlib.use('Agg')                      # no test should open a window
from types import SimpleNamespace
import numpy as np
import pytest

SEED = 0


# ------------------------------------------------------------ matrix generators

def spd(n, cond=50.0, seed=SEED):
    Q = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, n)))[0]
    A = Q @ np.diag(np.logspace(0, np.log10(cond), n)) @ Q.T
    return 0.5 * (A + A.T)


def indefinite(n, seed=SEED):
    Q = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, n)))[0]
    d = np.linspace(-2.0, 5.0, n)
    d[np.isclose(d, 0.0)] = 0.5
    A = Q @ np.diag(d) @ Q.T
    return 0.5 * (A + A.T)


def hard_case(n=3, seed=SEED):
    """
    (g, B, radius) in the hard case: g has no component along the eigenvector of
    lambda_min, so no lambda >= -lambda_min puts p on the boundary and the
    solution needs the eigenvector correction p* = p + tau*z.
    """
    Q = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, n)))[0]
    d = np.arange(-1.0, n - 1.0)
    d[0] = -1.0
    B = 0.5 * (Q @ np.diag(d) @ Q.T + (Q @ np.diag(d) @ Q.T).T)
    coeffs = np.ones(n)
    coeffs[0] = 0.0                                   
    g = Q @ coeffs
    p_at_bound = Q @ np.where(d + 1.0 > 1e-12, -coeffs / (d + 1.0), 0.0)
    return g, B, 4.0 * np.linalg.norm(p_at_bound) + 1.0


# ---------------------------------------------------- trust-region subproblem ref

def subproblem_reference(g, B, radius, tol=1e-13, max_iter=300):
    """
    Exact solution of  min gp + 1/2 pBp  s.t.  ||p|| <= radius
    """
    d, V = np.linalg.eigh(B)
    q = V.T @ np.asarray(g, dtype=float)

    if d[0] > 0:                                       
        p = -V @ (q / d)
        if np.linalg.norm(p) <= radius:
            return p

    def p_norm(lam):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.linalg.norm(np.nan_to_num(q / (d + lam), posinf=np.inf))

    lo = max(0.0, -d[0]) + 1e-14
    hi = lo + 1.0
    while p_norm(hi) > radius:
        hi *= 2.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if p_norm(mid) > radius:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    lam = 0.5 * (lo + hi)

    p = -V @ np.where(np.abs(d + lam) > 1e-14, q / (d + lam), 0.0)
    norm = np.linalg.norm(p)
    if norm < radius * (1.0 - 1e-8):                   
        p = p + np.sqrt(max(radius**2 - norm**2, 0.0)) * V[:, 0]
    return p


def model_value(g, B, p):
    return g @ p + 0.5 * p @ B @ p


# --------------------------------------------------------------- test problems

def _quadratic(n=4, cond=50.0):
    A, b = spd(n, cond), np.ones(n)
    x_opt = np.linalg.solve(A, b)

    def f(x):
        return 0.5 * x @ A @ x - b @ x

    # *args: trust_region.optimize calls h(x_cur, g_cur), passing the gradient
    # as a second positional argument 
    f.gradient = lambda x, *args, **kwargs: A @ x - b
    f.hessian = lambda x, *args, **kwargs: A
    f.__name__ = 'quadratic'
    return SimpleNamespace(func=f, grad=f.gradient, hess=f.hessian, A=A, b=b,
                           n=n, x0=np.zeros(n), x_opt=x_opt, f_star=f(x_opt))


def _rosenbrock():
    from presto.test_functions import rosenbrock
    return SimpleNamespace(func=rosenbrock, grad=rosenbrock.gradient,
                           hess=rosenbrock.hessian, n=2,
                           x0=np.array([1.2, 1.2]), x0_hard=np.array([-1.2, 1.0]),
                           x_opt=np.ones(2), f_star=0.0)


def _rosenbrock_residual():
    a, b = 1.0, 100.0
    sb, sa = np.sqrt(2.0 * b), np.sqrt(2.0)

    def r(x):
        x = np.asarray(x, dtype=float)
        return np.array([sb * (x[1] - x[0] ** 2), sa * (a - x[0])])

    def jac(x):
        x = np.asarray(x, dtype=float)
        return np.array([[-2.0 * sb * x[0], sb], [-sa, 0.0]])

    r.jacobian = r.gradient = jac
    r.__name__ = 'rosenbrock_residual'
    return SimpleNamespace(func=r, jac=jac, n=2, m=2, x_opt=np.ones(2),
                           x0=np.array([1.2, 1.2]), x0_hard=np.array([-1.2, 1.0]))


def _exp_decay(m=40, theta=(2.5, 0.8, 0.5), noise=0.0):
    """y = A exp(-k t) + c.  m >> n = 3, to test Levenberg-Marquadt"""
    theta = np.asarray(theta, dtype=float)
    t = np.linspace(0.0, 6.0, m)
    model = lambda tt, th: th[0] * np.exp(-th[1] * tt) + th[2]
    y = model(t, theta)
    if noise:
        y = y + noise * np.random.default_rng(SEED).standard_normal(m)

    def r(th):
        return y - model(t, th)

    def jac(th):
        e = np.exp(-th[1] * t)
        return np.column_stack([-e, th[0] * t * e, -np.ones_like(t)])

    r.jacobian = r.gradient = jac
    r.__name__ = 'exp_decay_residual'
    return SimpleNamespace(func=r, jac=jac, t=t, y=y, model=model, m=m, n=3,
                           x_opt=theta, x0=np.array([1.0, 0.2, 0.0]))


def _linear_lstsq(m=30, n=6, noise=1e-3):
    """Hilbert-like A: overdetermined and ill-conditioned, so the normal
    equations (cond squared) are worse than QR/SVD."""
    i, j = np.arange(m)[:, None], np.arange(n)[None, :]
    A = 1.0 / (i + j + 1.0)
    x_true = np.ones(n)
    b = A @ x_true + noise * np.random.default_rng(SEED).standard_normal(m)
    return SimpleNamespace(A=A, b=b, m=m, n=n, x_true=x_true,
                           x_opt=np.linalg.lstsq(A, b, rcond=None)[0])


@pytest.fixture
def quadratic():
    return _quadratic()


@pytest.fixture
def rosen():
    return _rosenbrock()


@pytest.fixture
def rosen_residual():
    return _rosenbrock_residual()


@pytest.fixture
def decay():
    return _exp_decay()


@pytest.fixture
def linear():
    return _linear_lstsq()


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)
