import numpy as np
import matplotlib.pyplot as plt

from presto.linalg import condition_number

def _stack(iterations, key):
    return np.array([np.asarray(it[key], dtype=float) for it in iterations])

def plot_cg_convergence(A, b, result, x_star=None, spectrum='A',
                        spectrum_style='stem', log=np.log, ax=None, title=None):
    """
    Plot log of the squared energy-norm error  ||x_k - x*||_A^2 = e^T A e
    (the quantity CG minimises) vs iteration, and overlay the eigenvalue
    spectrum that governs the convergence rate.

    spectrum : 'A'          -> eigenvalues of A
               'preconditioned' -> eigenvalues of M^-1 A M^-T
    spectrum_style : 'stem' -> one vertical line per eigenvalue (default)
                     'hist' -> histogram showing eigenvalue density
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if x_star is None:
        x_star = np.linalg.solve(A, b)

    iters = result.iterations
    ks = np.array([it['iter'] for it in iters])
    x_hist = _stack(iters, 'x')

    E = x_hist - x_star
    energy_sq = np.einsum('ij,jk,ik->i', E, A, E)          # e^T A e
    energy_sq = np.maximum(energy_sq, np.finfo(float).tiny)  # guard log(0)
    y = log(energy_sq)
    log_name = 'log10' if log is np.log10 else 'log'

    # spectrum to overlay
    if spectrum == 'preconditioned' and getattr(result, 'preconditioner', None) is not None:
        M = np.asarray(result.preconditioner, dtype=float)
        M_inv = np.linalg.solve(M, np.identity(M.shape[0]))
        S = M_inv @ A @ M_inv.T
        spec_label = r'eigenvalues of $M^{-1}AM^{-T}$'
    else:
        S = A
        spec_label = r'eigenvalues of $A$'
    R = 0.5 * (S + S.T)  # symmetrise for safety
    cond = condition_number(R)
    eig = np.linalg.eigvalsh(R)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(ks, y, 'o-', color='tab:blue', label=rf'${log_name}\,\|x_k-x^*\|_A^2$')
    ax.set_xlabel('iteration $k$')
    ax.set_ylabel(rf'${log_name}\,\|x_k-x^*\|_A^2$', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.grid(True, alpha=0.3)

    ax_top = ax.figure.add_axes(ax.get_position(), frameon=False)
    ax_top.set_xlim(eig.min(), eig.max())
    if spectrum_style == 'hist':
        ax_top.hist(eig, bins=min(20, len(eig)), color='tab:red', alpha=0.3)
        spec_note = '  (spectrum density)'
    elif spectrum_style == 'stem':
        ax_top.set_ylim(0, 1.05)
        ax_top.stem(eig, np.ones_like(eig), linefmt='tab:red',
                    markerfmt='.', basefmt=' ')
        spec_note = '  (spectrum)'
    else:
        raise ValueError(f"spectrum_style must be 'stem' or 'hist', got {spectrum_style!r}")
    ax_top.yaxis.set_visible(False)
    ax_top.xaxis.set_ticks_position('top')
    ax_top.xaxis.set_label_position('top')
    ax_top.spines['top'].set_color('tab:red')
    ax_top.set_xlabel(spec_label + spec_note, color='tab:red')
    ax_top.tick_params(axis='x', colors='tab:red')

    n = A.shape[0]
    ndist = len(np.unique(np.round(eig, 8)))
    ax.set_title(title or f'CG energy-norm convergence  '
                          fr'(n={n}, $\kappa$={cond:.2e}, {ndist} distinct eigvals)')
    ax.legend(loc='upper right')
    return ax


def plot_nonlinear_cg_convergence(result, f_min=None, log=np.log,
                              axes=None, title=None):
    """
    For nonlinear minimize there is no A / x*, so plot the two standard
    convergence diagnostics vs iteration:
        (a) log ||grad f(x_k)||
        (b) log (f(x_k) - f*)   (optimality gap; f* defaults to min f seen)
    """
    iters = result.iterations
    ks = np.array([it['iter'] for it in iters])
    grads = _stack(iters, 'grad')
    fvals = np.array([float(it['func']) for it in iters])

    gnorm = np.linalg.norm(grads, axis=1)
    gnorm = np.maximum(gnorm, np.finfo(float).tiny)

    if f_min is None:
        f_min = fvals.min()
    gap = np.maximum(fvals - f_min, np.finfo(float).tiny)

    log_name = 'log10' if log is np.log10 else 'log'
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 5))
    a0, a1 = axes

    a0.plot(ks, gnorm, 'o-', color='tab:green')
    a0.set_xlabel('iteration $k$')
    a0.set_ylabel(rf'${log_name}\,\|\nabla f(x_k)\|$')
    #a0.set_ylabel(rf'$gradient norm\,\|\nabla f(x_k)\|$')

    a0.grid(True, alpha=0.3)
    a0.set_title('gradient-norm convergence')

    a1.plot(ks, log(gap), 'o-', color='tab:purple')
    a1.set_xlabel('iteration $k$')
    a1.set_ylabel(rf'${log_name}\,(f(x_k)-f^*)$')
    a1.grid(True, alpha=0.3)
    a1.set_title('optimality gap')

    method = getattr(result, 'conjugate_method', None)
    method = method.__name__ if callable(method) else method
    if title or method:
        axes[0].figure.suptitle(title or f'nonlinear CG ({method}) convergence')
    return axes
