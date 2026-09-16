import matplotlib.pyplot as plt
from functools import partial
from matplotlib.colors import LogNorm
import numpy as np 
from presto.linalg import l2_norm

def evaluate(f, x_vals): # or batch
    return np.array([f(x) for x in x_vals]) # vectorise

def evaluate_func_and_gradient(func, x_vals, *args, **kwargs):
    grad = func.gradient

    f = partial(func, *args, **kwargs)
    g = partial(grad, *args, **kwargs)

    return evaluate(f, x_vals), evaluate(g, x_vals) 

def plot_function(func, min_val = -2, max_val = 3, num=50, n=2, grad=False, *args, **kwargs):
    coords = np.stack(
        np.meshgrid(*([np.linspace(min_val, max_val, num)]*n), indexing='ij'), 
        axis=0
        ).reshape(n, -1)
    X = coords.T   
    vals = np.array([func(v, *args, **kwargs) for v in coords.T]).reshape(coords.shape[1:])  

    vals, grad_vals = evaluate_func_and_gradient(func, X, *args, **kwargs)
    vals, grad_vals = vals.reshape(coords.shape[1:]), grad_vals.T
    
    if n == 2:
        plt.scatter(coords[0], coords[1], c=vals, cmap="viridis")
        if grad:
            plt.quiver(coords[0], coords[1],
                    grad_vals[0], grad_vals[1],
                    color="white", alpha=0.6)
        plt.colorbar(label="f(x)")
        plt.xlabel("x1")
        plt.ylabel("x2")
    if n==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(coords[0], coords[1], coords[2],
                        c=vals, cmap="viridis")
        if grad:
            ax.quiver(coords[0], coords[1], coords[2],
                grad_vals[0], grad_vals[1], grad_vals[2],
                length=0.1, normalize=True, color="white")
        fig.colorbar(sc, label="f(x)")

    if n > 3:
        from sklearn.decomposition import PCA

        proj = PCA(n_components=2).fit_transform(coords.T)
        plt.scatter(proj[:,0], proj[:,1], c=vals, cmap="viridis")
        plt.colorbar(label="f(x)")

def draw_numbered_path(ax, points, line_color="white", point_color="red"):
    points = np.asarray(points)
    for x_from, x_to in zip(points[:-1], points[1:]):
        if np.allclose(x_from, x_to):
            continue                          
        ax.annotate(
            "", xy=x_to, xytext=x_from,
            arrowprops=dict(arrowstyle="->", color=line_color, lw=1.5,
                            shrinkA=6, shrinkB=6),
            zorder=3,
        )
    ax.scatter(points[:, 0], points[:, 1], c=point_color,
               edgecolors="white", s=60, zorder=4)
    for k, (px, py) in enumerate(points):
        ax.annotate(
            f"{str(k)}", xy=(px, py), xytext=(5, 5),
            textcoords="offset points", fontsize=10, color=line_color,
            weight="bold", ha="left", va="bottom", zorder=5, clip_on=False,
        )

def name_of(obj, default="f"):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, partial):
        obj = obj.func
    return getattr(obj, "__name__", default)

def plot_contour(func, x_vals, direction=None, method=None, func_name=None, n=2, *args, **kwargs):
    if n != 2:
        return

    fname = name_of(func) if func_name is None else func_name
    dname = name_of(direction) if direction is not None else ""
    mname = name_of(method) if method is not None else ""
    suffix = f"{fname}" + (f" — {dname} with {mname}" if dname and mname else "")

    x1_min, x1_max = x_vals[:, 0].min() - 0.5, x_vals[:, 0].max() + 0.5
    x2_min, x2_max = x_vals[:, 1].min() - 0.5, x_vals[:, 1].max() + 0.5

    x1 = np.linspace(x1_min, x1_max, 500)
    x2 = np.linspace(x2_min, x2_max, 500)
    X1, X2 = np.meshgrid(x1, x2)

    f = partial(func, *args, **kwargs)

    Z = np.array([
        [f(np.array([xx, yy])) for xx, yy in zip(row_x, row_y)]
        for row_x, row_y in zip(X1, X2)
    ])

    fig, ax = plt.subplots(figsize=(8, 6))

    if Z.min() + 1e-8 > 0:
        l, h = np.log10(Z.min() + 1e-8), np.log10(Z.max())
        levels_f=np.logspace(l, h, 80)
        levels=np.logspace(l, h, 40)
    else:
        l, h = Z.min() + 1e-8, Z.max()
        levels_f=np.linspace(l, h, 80)
        levels=np.linspace(l, h, 40)

    cf = ax.contourf(
        X1, X2, Z,
        levels=levels_f,
        norm=LogNorm(),
        cmap="viridis",
    )

    ax.contour(
        X1, X2, Z,
        levels=levels,
        colors="black",
        alpha=0.3,
        linewidths=0.6,
    )

    draw_numbered_path(ax, x_vals, line_color="white", point_color="red")

    fig.colorbar(cf, ax=ax, label="f(x)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(suffix)

    plt.pause(0.001)

def plot_iterations(func_vals, grad_vals, search_directions, func_name=None, method=None, direction=None):
    n = len(func_vals)
    grad_vals = np.asarray(grad_vals)
    search_directions = np.asarray(search_directions)

    fname = name_of(func_name) if func_name is not None else ""
    dname = name_of(direction) if direction is not None else ""
    mname = name_of(method) if method is not None else ""

    suffix = (f" — {fname} —" if fname else "") + (f" {dname} with" if dname else "") + (f" {mname}" if mname else "")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 10))

    ax1.semilogy(range(n), func_vals, marker="o")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("f(x)")
    ax1.set_title("Function value per iteration")
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    ax2.plot(range(n), [l2_norm(g) for g in grad_vals], marker="o")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Gradient norm")
    ax2.set_title("Gradient norm per iteration")
    ax2.grid(True, which="both", ls="--", alpha=0.3)

    draw_numbered_path(ax3, grad_vals, line_color="black", point_color="tab:blue")
    ax3.axhline(0, color="gray", lw=0.7)
    ax3.axvline(0, color="gray", lw=0.7)
    ax3.set_xlabel(r"$\nabla f_1$")
    ax3.set_ylabel(r"$\nabla f_2$")
    ax3.set_title("Gradient progression")
    ax3.grid(True, ls="--", alpha=0.3)

    draw_numbered_path(ax4, search_directions, line_color="black", point_color="tab:green")
    ax4.axhline(0, color="gray", lw=0.7)
    ax4.axvline(0, color="gray", lw=0.7)
    ax4.set_xlabel(r"$p_1$")
    ax4.set_ylabel(r"$p_2$")
    ax4.set_title("Search directions")
    ax4.grid(True, ls="--", alpha=0.3)

    fig.suptitle("Convergence" + suffix)
    fig.tight_layout()
    plt.pause(0.001)