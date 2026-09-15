"""
Trust-region subproblem solvers and the model machinery around them.

The exact solvers are checked against conftest.subproblem_reference (independent
eigen-expansion + bisection); the approximate ones against the two properties
that make them usable at all - stay inside the region, and achieve at least the
Cauchy-point decrease.
"""
import numpy as np
import pytest

from conftest import (hard_case, indefinite, model_value, spd,
                      subproblem_reference)
from presto.trust_region.trust_region import (cauchy_point,
                                              cholesky_trust_region_subproblem,
                                              dogleg, generalised_cauchy_point,
                                              qr_trust_region_subproblem,
                                              quadratic_model, reduction_ratio,
                                              solve_quadratic)

RTOL = 1e-6


def cases():
    """(id, g, B, radius) over the three regimes, each with an active and an
    inactive constraint."""
    n = 4
    g = np.arange(1.0, n + 1)
    out = []
    for label, B in (('pd', spd(n)), ('indefinite', indefinite(n))):
        p_newton = np.linalg.solve(B, -g) if label == 'pd' else None
        big = 10.0 * np.linalg.norm(g)
        small = 0.1
        out.append((f'{label}-boundary', g, B, small))
        out.append((f'{label}-interior', g, B, big))
        if p_newton is not None:
            out.append((f'{label}-exact-newton', g, B,
                        2.0 * np.linalg.norm(p_newton)))
    g_h, B_h, rad_h = hard_case()
    out.append(('hard', g_h, B_h, rad_h))
    return out


CASES = cases()
IDS = [c[0] for c in CASES]


# ----------------------------------------------------------------- exact solvers

@pytest.mark.parametrize('name,g,B,rad', CASES, ids=IDS)
def test_cholesky_matches_reference(name, g, B, rad):
    p = cholesky_trust_region_subproblem(g, B, rad)
    assert np.linalg.norm(p) <= rad * (1 + 1e-6), "step left the trust region"
    m, m_ref = model_value(g, B, p), model_value(g, B, subproblem_reference(g, B, rad))
    assert m <= m_ref + RTOL * max(1.0, abs(m_ref)), \
        f"model value {m:.6e} worse than the exact optimum {m_ref:.6e}"


@pytest.mark.parametrize('name,g,B,rad', CASES, ids=IDS)
def test_cholesky_step_is_a_descent_step(name, g, B, rad):
    """m(p) < m(0) whenever g != 0 - the property `general` relies on for rho."""
    p = cholesky_trust_region_subproblem(g, B, rad)
    assert model_value(g, B, p) < 0.0


@pytest.mark.parametrize('rad', [0.05, 0.5, 5.0])
def test_qr_matches_reference(rad, linear):
    """
    qr_trust_region_subproblem takes (r, J) and models 0.5||r + Jp||^2, so the
    equivalent (g, B) for the reference is (J'r, J'J).
    """
    J, r = linear.A, linear.b
    p = qr_trust_region_subproblem(r, J, rad)
    g, B = J.T @ r, J.T @ J
    assert np.linalg.norm(p) <= rad * (1 + 1e-6)
    m, m_ref = model_value(g, B, p), model_value(g, B, subproblem_reference(g, B, rad))
    assert m <= m_ref + RTOL * max(1.0, abs(m_ref))


# ----------------------------------------------------------- approximate solvers

@pytest.mark.parametrize('name,g,B,rad', CASES, ids=IDS)
def test_cauchy_point_inside_region_and_decreases(name, g, B, rad):
    p = cauchy_point(g, B, rad)
    assert np.linalg.norm(p) <= rad * (1 + 1e-10)
    assert model_value(g, B, p) < 0.0


@pytest.mark.parametrize('name,g,B,rad', CASES, ids=IDS)
def test_dogleg_inside_region_and_beats_cauchy(name, g, B, rad):
    """
    Dogleg is only defined for PD B; on indefinite B it must fall back to
    steepest descent rather than return a nonsense step. Either way it should
    not do worse than the Cauchy point.
    """
    p = dogleg(g, B, rad)
    assert np.linalg.norm(p) <= rad * (1 + 1e-6)
    m_cauchy = model_value(g, B, cauchy_point(g, B, rad))
    assert model_value(g, B, p) <= m_cauchy + RTOL * max(1.0, abs(m_cauchy))


@pytest.mark.parametrize('rad', [0.1, 1.0, 10.0])
def test_generalised_cauchy_respects_elliptical_region(rad):
    """With scaling D the constraint is ||Dp|| <= radius, not ||p||."""
    n = 4
    g, B = np.arange(1.0, n + 1), spd(n)
    D = np.diag(np.linspace(1.0, 4.0, n))
    p = generalised_cauchy_point(g, B, rad, D)
    assert np.linalg.norm(D @ p) <= rad * (1 + 1e-8), \
        f"||Dp|| = {np.linalg.norm(D @ p):.4e} exceeds radius {rad}"


# ------------------------------------------------------------ model / helpers

def test_quadratic_model_matches_explicit_expansion():
    n = 4
    g, B = np.arange(1.0, n + 1), spd(n)
    p = np.linspace(-1.0, 1.0, n)
    f = 3.0
    expected = f + g @ p + 0.5 * p @ B @ p
    assert np.isclose(quadratic_model(f, g, B, p), expected)


def test_quadratic_model_is_exact_on_a_quadratic(quadratic):
    """For f = 1/2 x'Ax - b'x the model is the function, so m(p) == f(x+p)."""
    x = np.full(quadratic.n, 0.3)
    p = np.linspace(-0.2, 0.2, quadratic.n)
    f, g, B = quadratic.func(x), quadratic.grad(x), quadratic.hess(x)
    assert np.isclose(quadratic_model(f, g, B, p), quadratic.func(x + p))


@pytest.mark.parametrize('a,b,c', [(1.0, -3.0, 2.0), (2.0, 5.0, -3.0),
                                   (1.0, 0.0, -4.0), (0.5, -2.0, 1.5)])
def test_solve_quadratic_returns_actual_roots(a, b, c):
    hi, lo = solve_quadratic(a, b, c)
    assert hi >= lo
    for root in (hi, lo):
        assert np.isclose(a * root**2 + b * root + c, 0.0, atol=1e-10), \
            f"{root} is not a root of {a}x^2 + {b}x + {c}"


def test_solve_quadratic_matches_numpy_roots():
    a, b, c = 1.0, -5.0, 6.0
    assert np.allclose(sorted(solve_quadratic(a, b, c)),
                       sorted(np.roots([a, b, c])))


def test_solve_quadratic_rejects_complex_roots():
    with pytest.raises(ValueError):
        solve_quadratic(1.0, 0.0, 1.0)


@pytest.mark.parametrize('f_cur,f_next,m_next,expected', [
    (10.0, 8.0, 8.0, 1.0),          # model predicted the step exactly
    (10.0, 9.0, 8.0, 0.5),          # half the predicted decrease
    (10.0, 12.0, 8.0, -1.0),        # step made things worse
])
def test_reduction_ratio(f_cur, f_next, m_next, expected):
    assert np.isclose(reduction_ratio(f_cur, f_next, m_next), expected)


def test_reduction_ratio_guards_zero_predicted_decrease():
    """
    A model predicting no decrease must not divide by zero. The sign convention
    is a design choice: a step that improves f gets +inf (accept and expand), a
    step that worsens it gets 0 (reject). Note the exact tie f_next == f_cur
    currently also returns +inf, so a zero-progress step is accepted.
    """
    assert reduction_ratio(10.0, 9.0, 10.0) == np.inf       # improved
    assert reduction_ratio(10.0, 11.0, 10.0) == 0.0         # worsened
    assert np.isfinite(reduction_ratio(10.0, 11.0, 10.0))
