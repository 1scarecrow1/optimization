import argparse
from functools import partial
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from presto.plotting.conjugate_gradient import plot_nonlinear_cg_convergence
from presto.least_squares.linear import LINEAR_LSTSQ_METHODS
from presto.least_squares.nonlinear import gauss_newton, levenberg_marquardt
from presto.plotting.least_squares import plot_fit, plot_lstsq_comparison, plot_residuals
from presto.line_search.line_search import backtracking
from presto.linalg import condition_number, l2_norm
from presto.plotting import name_of, plot_contour, plot_iterations
from presto.reporting import attempt
from presto.test_functions import rosenbrock
from presto.trust_region.trust_region import (cholesky_trust_region_subproblem,
                                              dogleg, qr_trust_region_subproblem)

plt.ion()


# ---------------------------------------------------------------- test problems

def rosenbrock_residual(x, a=1.0, b=100.0):
    """
    Residual form of rosenbrock, chosen so that 0.5‖r‖² == rosenbrock(x, a, b)
    exactly (n = 2). Lets a least-squares run be plotted on, and compared
    against, the same contour the line_search / trust_region runners use.
    """
    x = np.asarray(x, dtype=float)
    return np.array([np.sqrt(2.0 * b) * (x[1] - x[0] ** 2),
                     np.sqrt(2.0) * (a - x[0])])


def rosenbrock_residual_jac(x, a=1.0, b=100.0):
    x = np.asarray(x, dtype=float)
    return np.array([[-2.0 * np.sqrt(2.0 * b) * x[0], np.sqrt(2.0 * b)],
                     [-np.sqrt(2.0),                  0.0]])


rosenbrock_residual.jacobian = rosenbrock_residual_jac
rosenbrock_residual.gradient = rosenbrock_residual_jac   # gradient_func looks for .gradient


def exp_decay(t, theta):
    A, k, c = theta
    return A * np.exp(-k * t) + c


def make_decay_problem(m=40, theta_true=(2.5, 0.8, 0.5), noise=0.02, seed=0):
    """y = A·exp(-k t) + c + epsilon.  m >> n"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 6.0, m)
    y = exp_decay(t, np.asarray(theta_true, dtype=float)) + noise * rng.standard_normal(m)

    def residual(theta):                        # r(θ) = y − f(t; θ)
        return y - exp_decay(t, theta)

    def jacobian(theta):                        # ∂r/∂θ, (m, 3), analytic
        A, k, _ = theta
        e = np.exp(-k * t)
        return np.column_stack([-e, A * t * e, -np.ones_like(t)])

    residual.jacobian = jacobian
    residual.gradient = jacobian
    return SimpleNamespace(t=t, y=y, residual=residual, jacobian=jacobian, model=exp_decay,
                           theta_true=np.asarray(theta_true, dtype=float),
                           theta0=np.array([1.0, 0.2, 0.0]))


def make_linear_problem(m=30, n=6, seed=0):
    """Hilbert-like ill-conditioned matrix"""
    rng = np.random.default_rng(seed)
    i = np.arange(m)[:, None]
    j = np.arange(n)[None, :]
    A = 1.0 / (i + j + 1.0)
    x_true = np.ones(n)
    b = A @ x_true + 1e-3 * rng.standard_normal(m)
    return A, b, x_true


# -------------------------------------------------------------------- summarise results

def summarise_function(residual, x, jac=None):
    r = residual(x)
    J = (jac or getattr(residual, 'jacobian', None))(x)
    return SimpleNamespace(**{"At point": x, "residual": r, "0.5||r||^2": 0.5 * (r @ r),
                              "jacobian": J, "grad": J.T @ r, "cond(J)": condition_number(J)})


def summarise_search(result, display_all=False, save_results=False):
    solver = name_of(result.solver)
    method = name_of(getattr(result, 'search_method', None) or '')
    print(f"Minimising {name_of(result.func)} with {solver}" + (f" - {method}" if method else ""))
    print(f"min value: {float(result.f_min):.8g}, at {result.x}, "
          f"found in {len(result.iterations) - 1} iterations")

    if display_all:
        keys = [k for k in ('iter', 'step', 'size', 'x', 'func', 'grad')
                if k in result.iterations[0]]
        for it in result.iterations:
            print(" | ".join(f"{k}: {it[k]}" for k in keys))

    if save_results:
        pass


def _stack(iterations, key):
    return np.array([np.atleast_1d(np.asarray(it[key], dtype=float)) for it in iterations])


def plot_results(result, func=None, func_name=None, save_results=False):
    its = result.iterations
    func = func if func is not None else result.func
    func_name = func_name or name_of(func)
    method = name_of(getattr(result, 'search_method', None) or result.solver)

    x_vals = _stack(its, 'x')
    func_vals = np.array([float(it['func']) for it in its])
    grad_vals = _stack(its, 'grad')

    # gauss_newton stores 'step' as the scalar alpha and the vector in
    # 'search_direction'; levenberg_marquardt stores the vector in 'step'.
    key = next(k for k in ('search_direction', 'step', 'direction') if k in its[0])
    steps = _stack(its, key)

    if x_vals.shape[1] == 2 and steps.shape[1] == 2:
        plot_contour(func, x_vals, '', method, func_name=func_name)
        plot_iterations(func_vals, grad_vals, steps, func_name=func_name, method=method)
    else:
        plot_nonlinear_cg_convergence(result, title=f'{func_name} — {method}')

    if save_results:
        pass


def run_linear(m=30, n=6):
    A, b, x_true = make_linear_problem(m, n)
    print(f"A {A.shape}, cond(A) = {condition_number(A):.3e}")
    x_ref = np.linalg.lstsq(A, b, rcond=None)[0]

    header = f"{'method':<10}{'||Ax-b||':>14}{'||x-x_ref||':>14}{'||x-x_true||':>14}"
    print(header)
    print(f"{'lstsq':<10}{l2_norm(A @ x_ref - b):>14.4e}{0.0:>14.4e}{l2_norm(x_ref - x_true):>14.4e}")

    solutions = {'lstsq': x_ref}
    for name, solve in LINEAR_LSTSQ_METHODS.items():
        try:
            x = solve(A, b, np.zeros(n)) if name == 'cg' else solve(A, b)
        except Exception as exc:
            print(f"{name:<10}  failed: {type(exc).__name__}: {exc}")
            continue
        solutions[name] = np.asarray(x, dtype=float)
        print(f"{name:<10}{l2_norm(A @ x - b):>14.4e}"
              f"{l2_norm(x - x_ref):>14.4e}{l2_norm(x - x_true):>14.4e}")

    plot_lstsq_comparison(A, b, solutions)
    return solutions


def run_gauss_newton(x0=(-1.2, 1.0), subproblem_method='qr', max_iter=200):
    print(summarise_function(rosenbrock_residual, np.asarray(x0, dtype=float)))
    res = gauss_newton(rosenbrock_residual, np.asarray(x0, dtype=float),
                       jac=rosenbrock_residual_jac,
                       loss_function='quadratic',
                       subproblem_method=subproblem_method,
                       line_search_method=backtracking,
                       line_search_args={'c1': 1e-4, 'rho': 0.5},
                       max_iter=max_iter)
    summarise_search(res, display_all=False)
    plot_results(res, func=rosenbrock, func_name='rosenbrock (via residuals)')
    return res


def run_levenberg_marquardt(x0=(-1.2, 1.0), trust_region_method=qr_trust_region_subproblem,
                            rad0=1.0, max_iter=200):
    res = levenberg_marquardt(rosenbrock_residual, np.asarray(x0, dtype=float),
                              jac=rosenbrock_residual_jac,
                              loss_function='quadratic',
                              trust_region_method=trust_region_method,
                              trust_region_args={'rad0': rad0},
                              max_iter=max_iter)
    summarise_search(res, display_all=False)
    plot_results(res, func=rosenbrock, func_name='rosenbrock (via residuals)')
    return res


def run_curve_fit(solver='lm', max_iter=200, **kwargs):
    p = make_decay_problem()
    print(f"fitting A·exp(-k t) + c to {p.t.size} points, true θ = {p.theta_true}")
    run = levenberg_marquardt if solver in ('lm', 'levenberg_marquardt') else gauss_newton
    if run is gauss_newton:
        kwargs.setdefault('line_search_method', backtracking)
        kwargs.setdefault('line_search_args', {'c1': 1e-4, 'rho': 0.5})

    res = run(p.residual, p.theta0, jac=p.jacobian,
              loss_function='quadratic', max_iter=max_iter, **kwargs)
    summarise_search(res, display_all=False)
    print(f"θ̂ = {res.x}\nθ* = {p.theta_true}\n||θ̂ - θ*|| = {l2_norm(res.x - p.theta_true):.4e}")

    plot_fit(p.t, p.y, p.model, res.x, theta_true=p.theta_true,
             iterations=res.iterations, title='exponential decay fit')
    plot_residuals(p.t, p.residual(res.x), title='exponential decay fit')
    plot_results(res, func_name='exp_decay')
    return res


def run_compare(x0=(-1.2, 1.0), max_iter=200):
    rows = []
    configs = [
        ('gauss_newton (qr)',  run_gauss_newton,        {'subproblem_method': 'qr'}),
        ('gauss_newton (svd)', run_gauss_newton,        {'subproblem_method': 'svd'}),
        ('lm (qr subproblem)', run_levenberg_marquardt, {'trust_region_method': qr_trust_region_subproblem}),
        ('lm (cholesky)',      run_levenberg_marquardt, {'trust_region_method': cholesky_trust_region_subproblem}),
        ('lm (dogleg)',        run_levenberg_marquardt, {'trust_region_method': dogleg}),
    ]
    for label, func, kw in configs:
        res = attempt(label, func, x0=x0, max_iter=max_iter, **kw)
        if res is not None:
            rows.append((label, len(res.iterations) - 1, float(res.f_min), l2_norm(res.x - 1.0)))

    print(f"\n{'method':<22}{'iters':>8}{'f_min':>14}{'||x-x*||':>14}")
    for label, nit, fmin, err in rows:
        print(f"{label:<22}{nit:>8}{fmin:>14.4e}{err:>14.4e}")
    return rows



def parse_x0(s):
    return np.array([float(v) for v in s.split(",")])


RUNS = {
    'linear': run_linear,
    'gauss_newton': run_gauss_newton,
    'gn': run_gauss_newton,
    'lm': run_levenberg_marquardt,
    'levenberg_marquardt': run_levenberg_marquardt,
    'curve_fit': run_curve_fit,
    'compare': run_compare,
}


def main():
    run_linear()
    run_gauss_newton()
    run_levenberg_marquardt()
    run_curve_fit()
    plt.show(block=True)


def run():
    parser = argparse.ArgumentParser(description="run presto.least_squares")
    parser.add_argument("--run", choices=sorted(RUNS), default="compare")
    parser.add_argument("--x0", type=parse_x0, default="-1.2,1.0")
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    func = RUNS[args.run]
    kwargs = {'max_iter': args.max_iter}
    if func is not run_linear:
        kwargs['x0'] = tuple(args.x0)
    func(**kwargs)
    if not args.no_plot:
        plt.show(block=True)


if __name__ == "__main__":
    main()
