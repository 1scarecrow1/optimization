"""
Least squares: the linear solvers against numpy's reference, and the nonlinear
drivers (Gauss-Newton, Levenberg-Marquardt) on problems whose answer is known.

The linear tests deliberately use an ill-conditioned overdetermined system,
because that is where the differences between the methods show up: forming the
normal equations squares the condition number, and any solver that does it twice
loses badly.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from presto.least_squares.linear import (cg_solve, cholesky_solve, qr_solve,
                                         svd_solve)
from presto.least_squares.nonlinear import gauss_newton, levenberg_marquardt
from presto.linalg import condition_number, l2_norm
from presto.solvers.modified_newton import gauss_newton_approx
from presto.trust_region.trust_region import (cholesky_trust_region_subproblem,
                                              qr_trust_region_subproblem)

TOL = 1e-5


# ------------------------------------------------------------------ linear least squares

@pytest.mark.parametrize('solver', [qr_solve, svd_solve, cholesky_solve],
                         ids=['qr', 'svd', 'cholesky'])
def test_linear_solver_matches_numpy_lstsq(solver, linear):
    x = solver(linear.A, linear.b)
    assert x.shape == (linear.n,), f"expected shape ({linear.n},), got {np.shape(x)}"
    assert np.allclose(x, linear.x_star, rtol=1e-4, atol=1e-6), \
        f"||x - x_lstsq|| = {l2_norm(x - linear.x_star):.3e}"


@pytest.mark.parametrize('solver', [qr_solve, svd_solve, cholesky_solve],
                         ids=['qr', 'svd', 'cholesky'])
def test_linear_solver_satisfies_the_normal_equations(solver, linear):
    """A'(Ax - b) = 0 is the defining property of the least squares solution."""
    x = solver(linear.A, linear.b)
    grad = linear.A.T @ (linear.A @ x - linear.b)
    assert l2_norm(grad) <= 1e-6 * max(1.0, l2_norm(linear.A.T @ linear.b)), \
        f"||A'(Ax-b)|| = {l2_norm(grad):.3e}"


@pytest.mark.parametrize('solver', [qr_solve, svd_solve, cholesky_solve],
                         ids=['qr', 'svd', 'cholesky'])
def test_linear_solver_is_exact_on_a_square_system(solver):
    A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    b = np.array([1.0, 2.0, 3.0])
    assert np.allclose(solver(A, b), np.linalg.solve(A, b), rtol=1e-8, atol=1e-10)


def test_cg_solve_on_normal_equations(linear):
    """cg_solve needs an SPD system, so it gets A'A x = A'b."""
    B, y = gauss_newton_approx(linear.A), linear.A.T @ linear.b
    x = cg_solve(B, y, np.zeros(linear.n), preconditioning=False,
                 conv_tol=1e-12, max_iter=500)
    assert np.allclose(x, linear.x_star, rtol=1e-3, atol=1e-5), \
        f"||x - x_lstsq|| = {l2_norm(np.asarray(x) - linear.x_star):.3e}"


def test_normal_equations_square_the_conditioning(linear):
    """
    Documents why cholesky_solve should be handed A, not A'A: passing the Gauss-
    Newton matrix in means the conditioning is squared a second time.
    """
    cond_A = condition_number(linear.A)
    cond_AtA = condition_number(gauss_newton_approx(linear.A))
    assert cond_AtA > 0.1 * cond_A ** 2


# --------------------------------------------------------------- gauss-newton

@pytest.mark.parametrize('subproblem', ['qr', 'svd'])
def test_gauss_newton_on_rosenbrock_residual(subproblem, rosen_residual):
    res = gauss_newton(rosen_residual.func, rosen_residual.x0,
                       jac=rosen_residual.jac, loss_function='quadratic',
                       subproblem_method=subproblem, max_iter=200)
    assert np.allclose(res.x, rosen_residual.x_star, atol=1e-4), \
        f"x = {res.x}, x* = {rosen_residual.x_star}"


def test_gauss_newton_fits_exact_data(decay):
    """Zero-noise data: the residual must go to (numerically) zero."""
    res = gauss_newton(decay.func, decay.x0, jac=decay.jac,
                       loss_function='quadratic', subproblem_method='qr',
                       max_iter=300)
    assert l2_norm(decay.func(res.x)) <= 1e-6, \
        f"||r|| = {l2_norm(decay.func(res.x)):.3e}, theta = {res.x}"


def test_gauss_newton_logs_the_terminal_state(rosen_residual):
    res = gauss_newton(rosen_residual.func, rosen_residual.x0,
                       jac=rosen_residual.jac, loss_function='quadratic',
                       subproblem_method='qr', max_iter=200)
    assert np.allclose(res.iterations[-1]['x'], res.x), \
        "history does not end at the returned x: the convergence exit path " \
        "does not record the final iterate"


# --------------------------------------------------------- levenberg-marquardt

@pytest.mark.parametrize('method', [qr_trust_region_subproblem], ids=['qr'])
def test_levenberg_marquardt_on_rosenbrock_residual(method, rosen_residual):
    res = levenberg_marquardt(rosen_residual.func, rosen_residual.x0,
                              jac=rosen_residual.jac, loss_function='quadratic',
                              trust_region_method=method,
                              trust_region_args={'rad0': 1.0}, max_iter=300)
    assert np.allclose(res.x, rosen_residual.x_star, atol=1e-4), \
        f"x = {res.x}, x* = {rosen_residual.x_star}"


def test_levenberg_marquardt_fits_exact_data(decay):
    res = levenberg_marquardt(decay.func, decay.x0, jac=decay.jac,
                              loss_function='quadratic',
                              trust_region_args={'rad0': 1.0}, max_iter=400)
    assert np.allclose(res.x, decay.x_star, rtol=1e-3, atol=1e-4), \
        f"theta = {res.x}, theta* = {decay.x_star}"


def test_levenberg_marquardt_logs_every_iteration(rosen_residual):
    """
    iterations must accumulate across the loop, and record the state on the
    convergence / zero-radius exit paths as well as after a step.
    """
    res = levenberg_marquardt(rosen_residual.func, rosen_residual.x0,
                              jac=rosen_residual.jac, loss_function='quadratic',
                              trust_region_args={'rad0': 1.0}, max_iter=50)
    assert len(res.iterations) > 2, \
        f"only {len(res.iterations)} entries logged: the history list is being " \
        "rebuilt inside the loop instead of appended to"
    assert np.allclose(res.iterations[-1]['x'], res.x)


def test_levenberg_marquardt_is_monotone(rosen_residual):
    """Rejected trust-region steps are not taken, so f must never increase."""
    res = levenberg_marquardt(rosen_residual.func, rosen_residual.x0,
                              jac=rosen_residual.jac, loss_function='quadratic',
                              trust_region_args={'rad0': 1.0}, max_iter=300)
    f = np.array([float(it['func']) for it in res.iterations])
    assert np.all(np.diff(f) <= 1e-8 * np.maximum(1.0, np.abs(f[:-1]))), \
        f"f increased: max rise {np.diff(f).max():.3e}"


@pytest.mark.parametrize('method', ['dogleg', 'cauchy_point'])
def test_levenberg_marquardt_rejects_g_b_subproblems(method, rosen_residual):
    """
    LM hands (residual, Jacobian) to the subproblem. dogleg and cauchy_point read
    that pair as (gradient, Hessian) and return a step that is not a descent
    direction — measured: 14 iterations, x never leaves x0, no error raised.
    The pairing must fail loudly, not silently do nothing.
    """
    from presto.run_optimizer import run_lm
    p = SimpleNamespace(kind='residual', name='r', func=rosen_residual.func,
                        jac=rosen_residual.jac, x0=rosen_residual.x0)
    with pytest.raises(ValueError, match='levenberg_marquardt'):
        run_lm(p, method=method)


def test_gauss_newton_and_lm_agree(rosen_residual):
    gn = gauss_newton(rosen_residual.func, rosen_residual.x0,
                      jac=rosen_residual.jac, loss_function='quadratic',
                      subproblem_method='qr', max_iter=300)
    lm = levenberg_marquardt(rosen_residual.func, rosen_residual.x0,
                             jac=rosen_residual.jac, loss_function='quadratic',
                             trust_region_args={'rad0': 1.0}, max_iter=300)
    assert np.allclose(gn.x, lm.x, atol=1e-3), f"GN {gn.x} vs LM {lm.x}"


# ------------------------------------------------------------- loss functions

@pytest.mark.parametrize('name', ['squared', 'quadratic', 'sq', 'absolute', 'abs'])
def test_loss_function_resolves_to_a_callable(name):
    from presto.least_squares.loss_functions import LOSS_FUNCTIONS, quadratic_loss
    from presto.utils import resolve_func
    f = resolve_func(name, LOSS_FUNCTIONS, 'loss function', quadratic_loss)
    assert callable(f), f"{name!r} resolved to {f!r}"


@pytest.mark.parametrize('name', sorted(set(('huber', 'absolute', 'quadratic'))))
def test_registered_losses_share_the_func_x_signature(name):
    """
    Everything in LOSS_FUNCTIONS is called as partial(loss, r)(x), so a registered
    loss must take (func, x, ...) and return a scalar.
    """
    from presto.least_squares.loss_functions import LOSS_FUNCTIONS
    r = lambda x: np.asarray(x, dtype=float) - 1.0
    value = LOSS_FUNCTIONS[name](r, np.array([0.4, -0.7]))
    assert value is not None, f"{name} returned None"
    assert np.ndim(value) == 0, f"{name} returned shape {np.shape(value)}, want a scalar"


def test_quadratic_loss_is_half_squared_norm(rosen_residual):
    from presto.least_squares.loss_functions import quadratic_loss
    x = np.array([0.4, -0.7])
    r = rosen_residual.func(x)
    assert np.isclose(quadratic_loss(rosen_residual.func, x), 0.5 * (r @ r))


def test_residual_form_matches_the_scalar_objective(rosen_residual):
    """0.5||r(x)||^2 == rosenbrock(x): the two problem statements must agree."""
    from presto.test_functions import rosenbrock
    for x in ([1.2, 1.2], [-1.2, 1.0], [0.0, 0.0], [1.0, 1.0]):
        x = np.asarray(x, dtype=float)
        r = rosen_residual.func(x)
        assert np.isclose(0.5 * (r @ r), rosenbrock(x))


def test_gauss_newton_gradient_equals_jt_r(rosen_residual):
    """grad of 0.5||r||^2 is J'r - the identity both drivers are built on."""
    x = np.array([0.4, -0.7])
    r, J = rosen_residual.func(x), rosen_residual.jac(x)
    from presto.gradients import finite_difference
    fd = finite_difference(lambda z: 0.5 * (rosen_residual.func(z) @ rosen_residual.func(z)),
                           x, central=True, eps=1e-6)
    assert np.allclose(J.T @ r, fd, rtol=1e-4, atol=1e-6)
