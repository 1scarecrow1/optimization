import matplotlib.pyplot as plt
import numpy as np
from presto.linalg import l2_norm


def plot_fit(t, y, model, theta, theta_true=None, iterations=None, n_trace=4,
             title=None, ax=None):
    """Plot fitted curve"""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ts = np.linspace(float(np.min(t)), float(np.max(t)), 400)

    if iterations:
        picks = np.unique(np.linspace(0, len(iterations) - 1, n_trace).astype(int))
        for k, idx in enumerate(picks[:-1]):
            ax.plot(ts, model(ts, iterations[idx]['x']), color='tab:orange',
                    alpha=0.20 + 0.15 * k, lw=1.0,
                    label='iterates' if k == 0 else None)

    ax.plot(t, y, 'o', ms=4, color='tab:blue', label='data')
    if theta_true is not None:
        ax.plot(ts, model(ts, theta_true), '--', color='gray', lw=1.2, label='true')
    ax.plot(ts, model(ts, theta), '-', color='tab:red', lw=2.0, label='fit')

    r = y - model(t, theta)
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    ax.set_title((title or 'least-squares fit') +
                 rf'   ($\frac{{1}}{{2}}\|r\|^2$ = {0.5 * r @ r:.3e})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.pause(0.001)
    return ax


def plot_residuals(t, r, title=None, axes=None):
    r = np.asarray(r, dtype=float)
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    a0, a1 = axes

    a0.stem(t, r, linefmt='tab:purple', markerfmt='.', basefmt='gray')
    a0.axhline(0.0, color='gray', lw=0.8)
    a0.set_xlabel('t')
    a0.set_ylabel(r'$r_i$')
    a0.set_title(rf'residuals   ($\|r\|$ = {l2_norm(r):.3e})')
    a0.grid(True, alpha=0.3)

    a1.hist(r, bins=min(20, max(5, r.size // 3)), color='tab:purple', alpha=0.7)
    a1.axvline(0.0, color='gray', lw=0.8)
    a1.set_xlabel(r'$r_i$')
    a1.set_title(f'residual distribution (mean {r.mean():.2e})')
    a1.grid(True, alpha=0.3)

    if title:
        a0.figure.suptitle(title)
    a0.figure.tight_layout()
    plt.pause(0.001)
    return axes


def plot_lstsq_comparison(A, b, solutions, reference='lstsq', axes=None):
    """‖Ax-b‖ per solver, along with deviation from reference solution."""
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    names = list(solutions)
    resid = [l2_norm(A @ solutions[n] - b) for n in names]
    x_ref = solutions.get(reference)

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    a0, a1 = axes

    a0.bar(names, resid, color='tab:blue', alpha=0.8)
    a0.set_yscale('log')
    a0.set_ylabel(r'$\|Ax-b\|_2$')
    a0.set_title('residual norm by solver')
    a0.tick_params(axis='x', rotation=30)
    a0.grid(True, axis='y', alpha=0.3)

    for n in names:
        if x_ref is not None and n != reference:
            a1.plot(solutions[n] - x_ref, 'o-', ms=4, label=n)
    a1.axhline(0.0, color='gray', lw=0.8)
    a1.set_xlabel('component')
    a1.set_ylabel(rf'$x_i - x_i^{{\rm {reference}}}$')
    a1.set_title(f'deviation from {reference}')
    a1.legend(fontsize=8)
    a1.grid(True, alpha=0.3)

    a0.figure.tight_layout()
    plt.pause(0.001)
    return axes
