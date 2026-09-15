"""
Shared reporting for the per-module run_optimizer scripts.

Every driver returns a MinimizeResult-shaped object - x, f_min, iterations
(list of dicts), func, plus one or two fields naming the method - but the field
names and the iteration keys differ per module, which is why each runner grew
its own near-identical copy of summarise_search / plot_results:

    module              solver field       method field           step key
    line_search         solver             line_search_method     step (+ alpha)
    trust_region        solver             trust_region_method    step (+ size)
    conjugate_gradient  conjugate_method   line_search_method     direction | step
    inexact_newton      -                  line_search_method     direction
    least_squares       solver             search_method          search_direction | step

Nothing here hard-codes those names: they are resolved by first-match lookup, so
a new module gets reporting for free and a renamed field degrades to a blank
label instead of an AttributeError.
"""
from types import SimpleNamespace

import numpy as np

from presto.conjugate_gradient.plotting import plot_nonlinear_cg_convergence
from presto.linalg import condition_number, l2_norm
from presto.plotting import name_of, plot_contour, plot_iterations

# Order matters: first match wins.
SOLVER_ATTRS = ('solver', 'conjugate_method')
METHOD_ATTRS = ('line_search_method', 'trust_region_method', 'search_method')
# 'step' is checked last because gauss_newton stores the scalar alpha there and
# the step vector in 'search_direction'.
DIRECTION_KEYS = ('step', 'step', 'step')
DISPLAY_KEYS = ('iter', 'alpha', 'step', 'size', 'x', 'func', 'grad')


# ------------------------------------------------------------------ resolution

def _first_attr(result, attrs):
    return next((getattr(result, a) for a in attrs
                 if getattr(result, a, None) is not None), None)


def labels(result):
    """(solver_name, method_name); '' for whichever the driver does not report."""
    return (name_of(_first_attr(result, SOLVER_ATTRS) or ''),
            name_of(_first_attr(result, METHOD_ATTRS) or ''))


def direction_key(iterations):
    """Iteration key holding the step vector, or None if the driver logs no step."""
    return next((k for k in DIRECTION_KEYS if k in iterations[0]), None)


def history(iterations, key):
    """(n_iter, dim) array of one iteration field."""
    return np.array([np.atleast_1d(np.asarray(it[key], dtype=float))
                     for it in iterations])


def as_scalar(v):
    """f_min is a vector in cg_solve (it returns the residual) - report its norm."""
    v = np.asarray(v, dtype=float)
    return float(v) if v.ndim == 0 else float(l2_norm(v))


# ------------------------------------------------------------------- summaries

def summarise_function(func, x, grad=None, hess=None, **func_args):
    """
    Function and derivative summary at x.

    Only uses derivatives that are passed in or attached to func; it will not
    fall back to finite differences, which would cost O(n) extra evaluations
    (and O(n^2) for the Hessian) for what is only a one-line printout.
    """
    x = np.asarray(x, dtype=float)
    out = {'at': x, 'func': func(x, **func_args)}

    g = grad or getattr(func, 'gradient', None) or getattr(func, 'jacobian', None)
    if g is not None:
        gx = np.asarray(g(x, **func_args), dtype=float)
        out['grad'], out['grad_norm'] = gx, l2_norm(gx)

    h = hess or getattr(func, 'hessian', None)
    if h is not None:
        hx = np.asarray(h(x, **func_args), dtype=float)
        out['hess'] = hx
        if hx.ndim == 2 and hx.shape[0] == hx.shape[1]:
            out['eigvals'] = np.linalg.eigvalsh(0.5 * (hx + hx.T))
            out['hess_cond'] = condition_number(hx)
    return SimpleNamespace(**out)


def summarise_search(result, display_all=False, keys=None, save_results=False):
    solver, method = labels(result)
    what = " - ".join(s for s in (solver, method) if s)
    func = getattr(result, 'func', None)
    func_name = name_of(func) if func is not None else 'quadratic'
    # direct solvers legitimately return iterations=None
    nit = len(result.iterations) - 1 if result.iterations else 0

    value = as_scalar(result.f_min)
    tag = 'min value' if np.ndim(result.f_min) == 0 else 'min ||f||'
    print(f"Minimising {func_name}" + (f" with {what}" if what else ""))
    print(f"{tag}: {value:.8g}, at {result.x}, found in {nit} iterations")

    if display_all and result.iterations:
        keys = keys or [k for k in DISPLAY_KEYS if k in result.iterations[0]]
        for it in result.iterations:
            print(" | ".join(f"{k}: {it[k]}" for k in keys))

    if save_results:
        pass

    return SimpleNamespace(func=func_name, solver=solver, method=method,
                           nit=nit, f_min=value, x=result.x)


def compare(runs, x_star=None, sort=True):
    """
    One table over several results - the 'compare 2+ methods' TODO the runners
    all carry. `runs` is {label: result} or an iterable of results; None entries
    (a run that raised) are skipped.
    """
    items = runs.items() if isinstance(runs, dict) else [(None, r) for r in runs]
    x_star = None if x_star is None else np.asarray(x_star, dtype=float)

    rows = []
    for label, res in items:
        if res is None:
            continue
        solver, method = labels(res)
        its = res.iterations or []
        rows.append({
            'method': label or " - ".join(s for s in (solver, method) if s) or '?',
            'iters': len(its) - 1 if its else 0,
            'f_min': as_scalar(res.f_min),
            'grad': l2_norm(np.asarray(its[-1]['grad'], dtype=float)) if its else np.nan,
            'err': np.nan if x_star is None
                   else l2_norm(np.asarray(res.x, dtype=float) - x_star),
        })
    if not rows:
        print("nothing to compare")
        return rows
    if sort:
        rows.sort(key=lambda r: (r['f_min'], r['iters']))

    w = max(len(r['method']) for r in rows + [{'method': 'method'}])
    err_col = f"{'|x-x*|':>13}" if x_star is not None else ""
    print(f"{'method':<{w}}{'iters':>8}{'f_min':>14}{'|grad|':>13}" + err_col)
    for r in rows:
        line = f"{r['method']:<{w}}{r['iters']:>8}{r['f_min']:>14.4e}{r['grad']:>13.3e}"
        print(line + (f"{r['err']:>13.3e}" if x_star is not None else ""))
    return rows


# -------------------------------------------------------------------- plotting

def plot_results(result, func=None, func_name=None, save_results=False):
    """
    n == 2 with a 2-D step: contour path plus the four-panel convergence figure.
    Anything else: gradient-norm and optimality-gap traces, which need no path.
    """
    its = result.iterations
    if not its:
        return                      # direct solvers have no iteration history
    func = func if func is not None else getattr(result, 'func', None)
    func_name = func_name or (name_of(func) if func is not None else 'f')
    solver, method = labels(result)

    x_vals = history(its, 'x')
    grad_vals = history(its, 'grad')
    func_vals = np.array([as_scalar(it['func']) for it in its])
    key = direction_key(its)
    steps = history(its, key) if key else None

    planar = (func is not None and x_vals.shape[1] == 2 and grad_vals.shape[1] == 2
              and steps is not None and steps.shape[1] == 2)
    if planar:
        plot_contour(func, x_vals, solver, method, func_name=func_name)
        plot_iterations(func_vals, grad_vals, steps, func_name=func_name,
                        method=method, direction=solver)
    else:
        plot_nonlinear_cg_convergence(
            result, title=" - ".join(s for s in (func_name, solver, method) if s))

    if save_results:
        pass


# ----------------------------------------------------------------------- driving

def parse_x0(s):
    return np.array([float(v) for v in s.split(",")])


def attempt(label, fn, *args, **kwargs):
    """
    Run fn, report and continue on failure. Runner scripts sweep several methods
    and one broken path should not take the rest of the sweep down with it.
    """
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"  !! {label} failed: {type(exc).__name__}: {exc}")
        return None


def report(result, func=None, func_name=None, display_all=False, plot=True,
           save_results=False):
    """summarise + plot in one call - what every runner's run_optimizer() did."""
    summary = summarise_search(result, display_all=display_all,
                               save_results=save_results)
    if plot:
        plot_results(result, func=func, func_name=func_name,
                     save_results=save_results)
    return summary
