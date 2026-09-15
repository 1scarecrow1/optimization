"""
One runner for every presto driver, replacing the four per-module
run_optimizer.py scripts.

    python -m presto.run_optimizer                        # sweep everything
    python -m presto.run_optimizer --run trust_region
    python -m presto.run_optimizer --run lm --problem decay --display
    python -m presto.run_optimizer --run line_search --problem rosenbrock --x0 -1.2,1.0

Problems and drivers are two independent registries. A problem declares its
`kind` (scalar objective / residual vector / linear system) and a driver
declares which kinds it accepts, so the sweep only pairs things that make
sense and nothing has to be special-cased at the call site.
"""
import argparse
from functools import partial
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from presto.conjugate_gradient.conjugate_gradient import fletcher_reeves, polak_ribiere
from presto.line_search.line_search import backtracking, wolfe
from presto.linalg import condition_number, l2_norm
from presto.reporting import attempt, compare, parse_x0, report, summarise_function
from presto.test_functions import rosenbrock

plt.ion()

BACKTRACKING_ARGS = {'a0': 1.0, 'c1': 1e-4, 'rho': 0.5}
WOLFE_ARGS = {'a0': 1.0, 'a_max': 10.0, 'c1': 1e-4, 'c2': 0.1,
              'max_iter': 25, 'zoom_iter': 25, 'interp_method': 'cubic'}


# ------------------------------------------------------------------- problems

def _rosenbrock():
    return SimpleNamespace(
        kind='scalar', name='rosenbrock', func=rosenbrock,
        grad=rosenbrock.gradient, hess=rosenbrock.hessian,
        x0=np.array([1.2, 1.2]), x_star=np.ones(2), n=2)


def _quadratic(n=4, cond=50.0):
    """f = ½xᵀAx − bᵀx with A SPD — every Newton-type method should finish in one step."""
    Q = np.linalg.qr(np.random.default_rng(0).standard_normal((n, n)))[0]
    A = Q @ np.diag(np.logspace(0, np.log10(cond), n)) @ Q.T
    A = 0.5 * (A + A.T)
    b = np.ones(n)

    def f(x, *_, **__):
        return 0.5 * x @ A @ x - b @ x

    f.__name__ = 'quadratic'
    f.gradient = lambda x, *_, **__: A @ x - b
    f.hessian = lambda x, *_, **__: A
    return SimpleNamespace(kind='scalar', name='quadratic', func=f,
                           grad=f.gradient, hess=f.hessian, A=A, b=b,
                           x0=np.zeros(n), x_star=np.linalg.solve(A, b), n=n)


def _rosenbrock_residual():
    """0.5‖r‖² == rosenbrock(x) exactly, so least squares is comparable with the
    scalar drivers on the same contour."""
    sb, sa = np.sqrt(200.0), np.sqrt(2.0)

    def r(x, *_, **__):
        x = np.asarray(x, dtype=float)
        return np.array([sb * (x[1] - x[0] ** 2), sa * (1.0 - x[0])])

    def jac(x, *_, **__):
        x = np.asarray(x, dtype=float)
        return np.array([[-2.0 * sb * x[0], sb], [-sa, 0.0]])

    r.__name__ = 'rosenbrock_residual'
    r.jacobian = r.gradient = jac
    return SimpleNamespace(kind='residual', name='rosenbrock_residual', func=r,
                           jac=jac, contour=rosenbrock, x0=np.array([1.2, 1.2]),
                           x_star=np.ones(2), m=2, n=2)


def _decay(m=40, theta=(2.5, 0.8, 0.5), noise=0.02, seed=0):
    """y = A·exp(−k t) + c.  m ≫ n = 3 — the shape LM is actually for."""
    theta = np.asarray(theta, dtype=float)
    t = np.linspace(0.0, 6.0, m)
    model = lambda tt, th: th[0] * np.exp(-th[1] * tt) + th[2]
    y = model(t, theta)
    if noise:
        y = y + noise * np.random.default_rng(seed).standard_normal(m)

    def r(th, *_, **__):
        return y - model(t, th)

    def jac(th, *_, **__):
        e = np.exp(-th[1] * t)
        return np.column_stack([-e, th[0] * t * e, -np.ones_like(t)])

    r.__name__ = 'exp_decay_residual'
    r.jacobian = r.gradient = jac
    return SimpleNamespace(kind='residual', name='exp_decay', func=r, jac=jac,
                           t=t, y=y, model=model, contour=None,
                           x0=np.array([1.0, 0.2, 0.0]), x_star=theta, m=m, n=3)


def _hilbert(m=30, n=6, noise=1e-3, seed=0):
    """Overdetermined and ill-conditioned: where QR/SVD beat the normal equations."""
    i, j = np.arange(m)[:, None], np.arange(n)[None, :]
    A = 1.0 / (i + j + 1.0)
    x_true = np.ones(n)
    b = A @ x_true + noise * np.random.default_rng(seed).standard_normal(m)

    def r(x, *_, **__):
        return A @ x - b

    r.__name__ = 'hilbert_residual'
    r.jacobian = r.gradient = lambda x, *_, **__: A
    return SimpleNamespace(kind='linear', name='hilbert', func=r, jac=r.jacobian,
                           A=A, b=b, x_true=x_true, x0=np.zeros(n), m=m, n=n,
                           x_star=np.linalg.lstsq(A, b, rcond=None)[0])


PROBLEMS = {
    'rosenbrock': _rosenbrock,
    'quadratic': _quadratic,
    'rosenbrock_residual': _rosenbrock_residual,
    'decay': _decay,
    'hilbert': _hilbert,
}


# --------------------------------------------------------------------- drivers

def run_line_search(p, solver='newton', method=backtracking, max_iter=500,
                    conv_tol=1e-5, solver_args=None, **kw):
    from presto.line_search.optimize import minimize
    args = BACKTRACKING_ARGS if method is backtracking else WOLFE_ARGS
    return minimize(p.func, p.x0, solver, method,
                    solver_args=solver_args or {}, line_search_args=dict(args),
                    conv_tol=conv_tol, max_iter=max_iter, **kw)


def run_trust_region(p, solver=None, method='cholesky', rad0=1.0, max_iter=500,
                     conv_tol=1e-5, solver_args=None, **kw):
    import presto.solvers.newton as nu
    from presto.trust_region.optimize import minimize
    return minimize(p.func, p.x0, solver or nu.newton, method,
                    solver_args=solver_args or {},
                    trust_region_args={'rad0': rad0},
                    conv_tol=conv_tol, max_iter=max_iter, **kw)


def run_cg(p, conjugate_method=polak_ribiere, method=backtracking,
           max_iter=500, conv_tol=1e-5, **kw):
    from presto.conjugate_gradient.optimize import minimize
    args = BACKTRACKING_ARGS if method is backtracking else WOLFE_ARGS
    return minimize(p.func, p.x0, line_search_method=method,
                    conjugate_method=conjugate_method,
                    line_search_args=dict(args),
                    conv_tol=conv_tol, max_iter=max_iter, **kw)


def run_newton_cg(p, method=backtracking, max_iter=300, conv_tol=1e-5, **kw):
    from presto.inexact_newton.newton_cg import minimize
    # newton_cg hardcodes a0=1 in its own partial, so passing a0 here collides
    args = {k: v for k, v in BACKTRACKING_ARGS.items() if k != 'a0'}
    return minimize(p.func, p.x0, line_search=True, line_search_method=method,
                    line_search_args=args,
                    conv_tol=conv_tol, max_iter=max_iter, **kw)


def run_gauss_newton(p, subproblem='qr', method=backtracking, max_iter=200, **kw):
    from presto.least_squares.nonlinear import gauss_newton
    args = BACKTRACKING_ARGS if method is backtracking else WOLFE_ARGS
    return gauss_newton(p.func, p.x0, jac=p.jac, loss_function='quadratic',
                        subproblem_method=subproblem,
                        line_search_method=method, line_search_args=dict(args),
                        max_iter=max_iter, **kw)


# LM builds its step from (r, J). Only the QR subproblem consumes that pair;
# cholesky needs (Jᵀr, JᵀJ) and dogleg silently produces a garbage step from it
# (measured: 14 iterations, x never leaves x0, no error raised).
LM_SUBPROBLEMS = ('qr', 'cholesky')


def run_lm(p, method='qr', rad0=1.0, max_iter=300, **kw):
    from presto.least_squares.nonlinear import levenberg_marquardt
    from presto.trust_region.trust_region import TRUST_REGION_METHODS
    key = method if isinstance(method, str) else getattr(method, '__name__', method)
    if not any(key.startswith(s) for s in LM_SUBPROBLEMS):
        raise ValueError(
            f"levenberg_marquardt cannot use {key!r}: it passes (residual, Jacobian) "
            f"to the subproblem, which only {' / '.join(LM_SUBPROBLEMS)} accept. "
            f"dogleg and cauchy_point read that pair as (g, B) and return nonsense.")
    return levenberg_marquardt(p.func, p.x0, jac=p.jac, loss_function='quadratic',
                               trust_region_method=TRUST_REGION_METHODS.get(key, method),
                               trust_region_args={'rad0': rad0},
                               max_iter=max_iter, **kw)


def run_linear(p, solver='QR', max_iter=200, conv_tol=1e-8, **kw):
    from presto.least_squares.linear import linear_least_squares
    return linear_least_squares(p.func, p.x0, jac=p.jac, solver=solver,
                                conv_tol=conv_tol, max_iter=max_iter, **kw)


DRIVERS = {
    'line_search': SimpleNamespace(run=run_line_search, kinds=('scalar',)),
    'trust_region': SimpleNamespace(run=run_trust_region, kinds=('scalar',)),
    'cg': SimpleNamespace(run=run_cg, kinds=('scalar',)),
    'newton_cg': SimpleNamespace(run=run_newton_cg, kinds=('scalar',)),
    'gn': SimpleNamespace(run=run_gauss_newton, kinds=('residual', 'linear')),
    'lm': SimpleNamespace(run=run_lm, kinds=('residual', 'linear')),
    'linear': SimpleNamespace(run=run_linear, kinds=('linear',)),
}

# (driver, label, kwargs) triples for the full sweep
SWEEP = [
    ('line_search', 'newton / backtracking', {'solver': 'newton'}),
    ('line_search', 'newton / wolfe', {'solver': 'newton', 'method': wolfe}),
    ('line_search', 'bfgs / backtracking', {'solver': 'bfgs', 'solver_args': {'inv': True}}),
    ('line_search', 'sr1 / backtracking', {'solver': 'sr1', 'solver_args': {'inv': True}}),
    ('trust_region', 'newton / cholesky', {'method': 'cholesky'}),
    ('trust_region', 'newton / dogleg', {'method': 'dogleg'}),
    ('trust_region', 'newton / cauchy', {'method': 'cauchy_point'}),
    ('cg', 'CG / polak-ribiere', {'conjugate_method': polak_ribiere}),
    ('cg', 'CG / fletcher-reeves', {'conjugate_method': fletcher_reeves}),
    ('newton_cg', 'newton-CG', {}),
    ('gn', 'gauss-newton / qr', {'subproblem': 'qr'}),
    ('gn', 'gauss-newton / svd', {'subproblem': 'svd'}),
    ('lm', 'levenberg-marquardt / qr', {'method': 'qr'}),
    ('linear', 'direct / QR', {'solver': 'QR'}),
    ('linear', 'direct / SVD', {'solver': 'SVD'}),
    ('linear', 'direct / Cholesky', {'solver': 'Cholesky'}),
]


# --------------------------------------------------------------------- running

def _contour(p):
    """Function to draw the contour against — residual problems borrow a scalar twin."""
    return getattr(p, 'contour', None) or (p.func if p.kind == 'scalar' else None)


def run_one(driver, problem, label=None, display=False, plot=True, **kw):
    p = PROBLEMS[problem]() if isinstance(problem, str) else problem
    d = DRIVERS[driver]
    if p.kind not in d.kinds:
        raise ValueError(f"{driver} takes {d.kinds} problems, {p.name} is {p.kind!r}")
    res = d.run(p, **kw)
    report(res, func=_contour(p), func_name=p.name, display_all=display, plot=plot)
    return res


def run_sweep(problem, display=False, plot=True):
    p = PROBLEMS[problem]()
    print(f"\n{'#' * 72}\n# {p.name}   kind={p.kind}  n={p.n}  x0={p.x0}\n{'#' * 72}")
    if p.kind == 'linear':
        print(f"  A {p.A.shape}, cond(A) = {condition_number(p.A):.3e}")
    elif p.kind == 'scalar' and getattr(p.func, 'gradient', None):
        print(summarise_function(p.func, p.x0))

    results = {}
    for driver, label, kw in SWEEP:
        if p.kind not in DRIVERS[driver].kinds:
            continue
        res = attempt(f"{p.name}: {label}", run_one, driver, p,
                      display=display, plot=plot, **kw)
        if res is not None:
            results[label] = res

    print(f"\n--- {p.name} ---")
    compare(results, x_star=getattr(p, 'x_star', None))
    return results


def main(problems=None, display=False, plot=True, block=True):
    for name in (problems or PROBLEMS):
        attempt(f"sweep: {name}", run_sweep, name, display=display, plot=plot)
    if plot and block:
        plt.show(block=True)


def cli():
    ap = argparse.ArgumentParser(description="drive every presto optimizer")
    ap.add_argument('--run', default='all',
                    choices=['all', *DRIVERS], help="driver, or 'all' to sweep")
    ap.add_argument('--problem', default=None, choices=list(PROBLEMS),
                    help='default: every problem the driver accepts')
    ap.add_argument('--x0', type=parse_x0, default=None)
    ap.add_argument('--max_iter', type=int, default=None)
    ap.add_argument('--display', action='store_true', help='print every iteration')
    ap.add_argument('--no_plot', action='store_true')
    args = ap.parse_args()

    plot = not args.no_plot
    if args.run == 'all':
        main(problems=[args.problem] if args.problem else None,
             display=args.display, plot=plot)
        return

    kinds = DRIVERS[args.run].kinds
    names = ([args.problem] if args.problem
             else [n for n, f in PROBLEMS.items() if f().kind in kinds])
    kw = {} if args.max_iter is None else {'max_iter': args.max_iter}
    for name in names:
        p = PROBLEMS[name]()
        if args.x0 is not None:
            p.x0 = args.x0
        attempt(f"{name}: {args.run}", run_one, args.run, p,
                display=args.display, plot=plot, **kw)
    if plot:
        plt.show(block=True)


if __name__ == '__main__':
    cli()
