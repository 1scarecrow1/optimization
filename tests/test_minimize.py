"""
The optimizer drivers, swept over (solver x globalisation strategy).

Every test asserts the contract a minimizer owes its caller rather than a
specific trajectory: the gradient norm comes down, f does not increase, the
iterate lands on the known minimiser, and the iteration log is well formed and
records the terminal state. That last one is what breaks silently when a loop
records history at the wrong place.
"""
import numpy as np
import pytest

import presto.solvers.newton as nu
import presto.solvers.quasi_newton as qn
from presto.conjugate_gradient.conjugate_gradient import (fletcher_reeves,
                                                          polak_ribiere)
from presto.line_search.line_search import backtracking, wolfe
from presto.linalg import l2_norm
from presto.trust_region.trust_region import (cauchy_point,
                                              cholesky_trust_region_subproblem,
                                              dogleg)

TOL = 1e-5
LS_ARGS = {'a0': 1.0, 'c1': 1e-4, 'rho': 0.5}
WOLFE_ARGS = {'a0': 1.0, 'a_max': 10.0, 'c1': 1e-4, 'c2': 0.9,
              'max_iter': 10, 'zoom_iter': 10, 'interp_method': 'cubic'}


# ------------------------------------------------------------------- assertions

def assert_iteration_log(result, keys=('iter', 'x', 'func', 'grad')):
    its = result.iterations
    assert its, "no iterations recorded"
    for k in keys:
        assert k in its[0], f"iteration dict is missing {k!r}"
    assert [it['iter'] for it in its] == sorted(it['iter'] for it in its), \
        "iteration counter is not monotonic"
    assert np.allclose(its[-1]['x'], result.x), \
        "last logged iterate differs from the returned x: the terminal state " \
        "was not recorded on the exit path"
    assert np.isclose(np.asarray(its[-1]['func'], dtype=float).ravel()[0],
                      np.asarray(result.f_min, dtype=float).ravel()[0]), \
        "last logged f differs from the returned f_min"


def assert_converged(result, x_star, tol=TOL, atol=1e-4):
    g_final = l2_norm(np.asarray(result.iterations[-1]['grad'], dtype=float))
    g_first = l2_norm(np.asarray(result.iterations[0]['grad'], dtype=float))
    assert g_final < g_first, f"gradient norm did not decrease ({g_first:.3e} -> {g_final:.3e})"
    assert g_final <= tol or np.allclose(result.x, x_star, atol=atol), \
        f"did not converge: |grad| = {g_final:.3e}, x = {result.x}, x* = {x_star}"


def assert_monotone_descent(result, slack=1e-8):
    """Line search and accepted trust-region steps must not increase f."""
    f = np.array([np.asarray(it['func'], dtype=float).ravel()[0]
                  for it in result.iterations])
    rise = np.maximum(np.diff(f), 0.0)
    assert np.all(rise <= slack * np.maximum(1.0, np.abs(f[:-1]))), \
        f"f increased along the path: max rise {rise.max():.3e}"


# ------------------------------------------------------------------ line search

LINE_SEARCH_GRID = [
    ('newton', backtracking, {}, LS_ARGS),
    ('modified_newton', backtracking, {}, LS_ARGS),
    ('bfgs', backtracking, {'inv': True}, LS_ARGS),
    ('sr1', backtracking, {'inv': True}, LS_ARGS),
    ('newton', wolfe, {}, WOLFE_ARGS),
    ('bfgs', wolfe, {'inv': True}, WOLFE_ARGS),
]
LS_IDS = [f'{s}-{m.__name__}' for s, m, _, _ in LINE_SEARCH_GRID]


@pytest.mark.parametrize('solver,ls,solver_args,ls_args', LINE_SEARCH_GRID, ids=LS_IDS)
def test_line_search_on_quadratic(solver, ls, solver_args, ls_args, quadratic):
    from presto.line_search.optimize import minimize
    res = minimize(quadratic.func, quadratic.x0, solver, ls,
                   solver_args=solver_args, line_search_args=ls_args,
                   conv_tol=TOL, max_iter=200)
    assert_iteration_log(res)
    assert_converged(res, quadratic.x_star)


@pytest.mark.parametrize('solver,ls,solver_args,ls_args', LINE_SEARCH_GRID, ids=LS_IDS)
def test_line_search_on_rosenbrock(solver, ls, solver_args, ls_args, rosen):
    from presto.line_search.optimize import minimize
    res = minimize(rosen.func, rosen.x0, solver, ls,
                   solver_args=solver_args, line_search_args=ls_args,
                   conv_tol=TOL, max_iter=500)
    assert_iteration_log(res)
    assert_converged(res, rosen.x_star)


def test_newton_solves_a_quadratic_in_one_step(quadratic):
    """The Newton step is exact for a quadratic, so alpha = 1 must finish it."""
    from presto.line_search.optimize import minimize
    res = minimize(quadratic.func, quadratic.x0, 'newton', backtracking,
                   line_search_args=LS_ARGS, conv_tol=TOL, max_iter=50)
    assert len(res.iterations) - 1 <= 2, \
        f"took {len(res.iterations) - 1} iterations on an exact quadratic"


def test_line_search_rejects_vector_objective(rosen_residual):
    from presto.line_search.optimize import minimize
    with pytest.raises(ValueError, match='scalar'):
        minimize(rosen_residual.func, rosen_residual.x0, 'newton', backtracking,
                 line_search_args=LS_ARGS)


# ----------------------------------------------------------------- trust region

TRUST_REGION_GRID = [
    (nu.newton, cholesky_trust_region_subproblem, {}),
    (nu.newton, dogleg, {}),
    (nu.newton, cauchy_point, {}),
    (qn.bfgs, cholesky_trust_region_subproblem, {}),
    (qn.bfgs, dogleg, {}),
    (qn.symmetric_rank_one, cholesky_trust_region_subproblem, {}),
    (qn.symmetric_rank_one, cauchy_point, {}),
]
TR_IDS = [f'{s.__name__}-{m.__name__}' for s, m, _ in TRUST_REGION_GRID]


@pytest.mark.parametrize('solver,method,solver_args', TRUST_REGION_GRID, ids=TR_IDS)
def test_trust_region_on_quadratic(solver, method, solver_args, quadratic):
    from presto.trust_region.optimize import minimize
    res = minimize(quadratic.func, quadratic.x0, solver, method,
                   solver_args=solver_args, trust_region_args={'rad0': 1.0},
                   conv_tol=TOL, max_iter=300)
    assert_iteration_log(res, keys=('iter', 'x', 'func', 'grad', 'size'))
    assert_converged(res, quadratic.x_star)


@pytest.mark.parametrize('solver,method,solver_args', TRUST_REGION_GRID, ids=TR_IDS)
def test_trust_region_on_rosenbrock(solver, method, solver_args, rosen):
    from presto.trust_region.optimize import minimize
    res = minimize(rosen.func, rosen.x0, solver, method,
                   solver_args=solver_args, trust_region_args={'rad0': 1.0},
                   conv_tol=TOL, max_iter=500)
    assert_iteration_log(res, keys=('iter', 'x', 'func', 'grad', 'size'))
    assert_converged(res, rosen.x_star)


def test_trust_region_radius_stays_positive_and_bounded(rosen):
    from presto.trust_region.optimize import minimize
    res = minimize(rosen.func, rosen.x0, nu.newton, cholesky_trust_region_subproblem,
                   trust_region_args={'rad0': 1.0},
                   solver_args={'rad_max': 10.0}, conv_tol=TOL, max_iter=200)
    radii = np.array([it['size'] for it in res.iterations], dtype=float)
    assert np.all(radii > 0.0), "trust region radius hit zero or went negative"
    assert np.all(radii <= 10.0 + 1e-12), f"radius exceeded rad_max: {radii.max()}"


def test_trust_region_step_stays_inside_the_radius(rosen):
    """||p_k|| <= Delta_{k-1} for every accepted step."""
    from presto.trust_region.optimize import minimize
    res = minimize(rosen.func, rosen.x0, nu.newton, cholesky_trust_region_subproblem,
                   trust_region_args={'rad0': 1.0}, conv_tol=TOL, max_iter=200)
    its = res.iterations
    for prev, cur in zip(its[:-1], its[1:]):
        assert l2_norm(np.asarray(cur['step'], dtype=float)) <= prev['size'] * (1 + 1e-6), \
            f"step {l2_norm(cur['step']):.4e} exceeded radius {prev['size']:.4e}"


def test_hessian_call_contract(quadratic):
    """
    trust_region.optimize builds h from hessian_func then calls h(x_cur, g_cur),
    so a Hessian written as hess(x) alone is called with an extra positional
    argument. This pins the contract that conftest works around.
    """
    from presto.gradients import hessian_func
    h = hessian_func(quadratic.func)
    H = h(quadratic.x0, quadratic.grad(quadratic.x0))
    assert np.allclose(H, quadratic.A)


# ------------------------------------------------------------ conjugate gradient

@pytest.mark.parametrize('conjugate_method', [polak_ribiere, fletcher_reeves],
                         ids=['polak_ribiere', 'fletcher_reeves'])
def test_nonlinear_cg_on_rosenbrock(conjugate_method, rosen):
    from presto.conjugate_gradient.optimize import minimize
    res = minimize(rosen.func, rosen.x0, line_search_method=wolfe,
                   conjugate_method=conjugate_method,
                   line_search_args=WOLFE_ARGS, conv_tol=TOL, max_iter=500)
    assert_iteration_log(res, keys=('iter', 'x', 'func', 'grad', 'step'))
    assert_converged(res, rosen.x_star)


@pytest.mark.parametrize('n', [5, 8, 12])
def test_linear_cg_solves_hilbert_system(n):
    """CG on an SPD system must reach the direct solution within n iterations."""
    from presto.conjugate_gradient.optimize import cg_solve
    c = np.arange(n)
    A = 1.0 / (c[:, None] + c[None, :] + 1.0)
    b = np.ones(n)
    res = cg_solve(A, b, np.zeros(n), preconditioning=False,
                   conv_tol=1e-10, max_iter=10 * n)
    # Hilbert is savagely ill-conditioned (cond ~ 1e16 by n=12) and CG's rate goes
    # as sqrt(cond), so an absolute target is unreachable. Assert the contraction
    # CG does guarantee instead.
    r0, r1 = l2_norm(b), l2_norm(A @ res.x - b)
    assert r1 <= 1e-3 * r0, f"residual only fell {r0:.3e} -> {r1:.3e}"
    assert r1 <= l2_norm(A @ np.linalg.lstsq(A, b, rcond=None)[0] - b) + 1e-3 * r0


@pytest.mark.parametrize('n', [5, 8])
def test_preconditioned_cg_matches_unpreconditioned_solution(n):
    from presto.conjugate_gradient.optimize import cg_solve
    c = np.arange(n)
    A = 1.0 / (c[:, None] + c[None, :] + 1.0)
    b = np.ones(n)
    res = cg_solve(A, b, np.zeros(n), preconditioning=True,
                   conv_tol=1e-10, max_iter=10 * n)
    assert np.allclose(res.x, np.linalg.solve(A, b), rtol=1e-4, atol=1e-6)


# -------------------------------------------------------------- inexact newton

def test_newton_cg_with_line_search(rosen):
    from presto.inexact_newton.newton_cg import minimize
    res = minimize(rosen.func, rosen.x0, line_search=True,
                   line_search_method=backtracking, conv_tol=TOL, max_iter=300)
    assert_iteration_log(res, keys=('iter', 'x', 'func', 'grad', 'direction'))
    assert_converged(res, rosen.x_star)


def test_newton_cg_inner_solver_matches_direct_solve(quadratic):
    """
    cg_line_search solves B p = -g inexactly; with a tight inner tolerance on an
    SPD B it must agree with the direct solve.
    """
    from presto.inexact_newton.newton_cg import cg_line_search
    g = quadratic.grad(quadratic.x0)
    p = cg_line_search(g, quadratic.A, tol_cur=1e-12, max_iter=200)
    assert np.allclose(p, np.linalg.solve(quadratic.A, -g), rtol=1e-5, atol=1e-8)


# ----------------------------------------------------------------- consistency

def test_drivers_agree_on_the_minimiser(rosen):
    """
    Different globalisation strategies, same problem: they must land in the same
    place. Catches a driver that 'converges' to the wrong point.
    """
    from presto.conjugate_gradient.optimize import minimize as cg_minimize
    from presto.line_search.optimize import minimize as ls_minimize
    from presto.trust_region.optimize import minimize as tr_minimize

    xs = [
        ls_minimize(rosen.func, rosen.x0, 'newton', backtracking,
                    line_search_args=LS_ARGS, conv_tol=TOL, max_iter=500).x,
        tr_minimize(rosen.func, rosen.x0, nu.newton, cholesky_trust_region_subproblem,
                    trust_region_args={'rad0': 1.0}, conv_tol=TOL, max_iter=500).x,
        cg_minimize(rosen.func, rosen.x0, line_search_method=backtracking,
                    conjugate_method=polak_ribiere, line_search_args={'a0': 1.0},
                    conv_tol=TOL, max_iter=500).x,
    ]
    for x in xs:
        assert np.allclose(x, rosen.x_star, atol=1e-3), f"{x} != {rosen.x_star}"
