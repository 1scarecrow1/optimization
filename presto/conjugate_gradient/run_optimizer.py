import numpy as np
import scipy
from presto.line_search.line_search import backtracking
from presto.test_functions import *
from presto.linalg import *
from presto.conjugate_gradient.optimize import *
import matplotlib.pyplot as plt
from presto.utils import timer
from presto.conjugate_gradient.plotting import * 
from types import SimpleNamespace
import argparse

plt.ion()

def summarise_function(func, x):
    fx = func(x)
    gx = func.gradient(x)
    hx = func.hessian(x)
    summary = {
        "At point": x,
        "func": round(fx, 6),
        "gradient": gx,
        "hessian": hx
    }
    return SimpleNamespace(**summary)

def summarise_search(result, display_all=False, save_results=False):
    func_name = result.func.__name__ if result.func else 'quadratic'
    method = result.conjugate_method
    method = method.__name__ if callable(method) else method
    ls = result.line_search_method
    ls = ls.__name__ if callable(ls) else ls
    print(f"Minimising {func_name} with CG ({method}) / line search: {ls}")
    print(f"min value: {round(result.f_min, 8)}, at {result.x}, "
          f"found in {len(result.iterations)-1} iterations")
    if display_all:
        keys = ['iter', 'direction', 'x', 'func', 'grad']   # real keys
        for it in result.iterations:
            print(" | ".join(f"{k}: {it[k]}" for k in keys))

def plot_results(result, func_name=None, save_results=False):
    func = result.func
    func_name = func.__name__ if func_name is None else func_name
    method = result.conjugate_method
    method = method.__name__ if callable(method) else method
    its = result.iterations
    x_vals    = np.array([it["x"] for it in its])
    func_vals = np.array([it["func"] for it in its])
    grad_vals = np.array([it["grad"] for it in its])
    directions = np.array([it["direction"] for it in its])   
    from presto.plotting import plot_contour, plot_iterations  
    plot_contour(func, x_vals, '', method, func_name=func_name)
    plot_iterations(func_vals, grad_vals, directions, func_name=func_name, method=method)

def run_optimizer(objective_func, x0, conjugate_method=polak_ribiere,
                  line_search_method=None, func_name=None, save_results=False, **params):
    res = minimize(objective_func, x0,
                   line_search_method=line_search_method,
                   conjugate_method=conjugate_method, **params)
    summarise_search(res, display_all=True, save_results=save_results)
    plot_results(res, func_name=func_name, save_results=save_results)

def parse_x0(s):
    return np.array([float(v) for v in s.split(",")])

def run_linear_cg():
    optimizer_params = {'conv_tol': 1e-6, 'max_iter':10000}

    n_dims = [5, 8, 12, 20]
    for n in n_dims:
        print(f'n = {n}')
        c = np.arange(n)
        xx, yy = np.meshgrid(c, c)
        A = 1 / (xx + yy + 1)
        print('Original matrix')
        print(f"eigvals: {np.flip(np.linalg.eigvalsh(A))}")
        print(f'condition no. {condition_number(A)}')
        b = np.ones(n)
        x0 = np.zeros(n)
        #res = cg_solve(A, b, x0, **optimizer_params)
        res = cg_solve(A, b, x0, preconditioning=True, preconditioner=incomplete_cholesky, **optimizer_params)

        M = res.preconditioner
        if M is not None and np.all(np.isfinite(M)):
            M_inv = inverse(M)
            #precond = M_inv @ A @ M_inv.T
            precond = M
            if np.all(np.isfinite(precond)):
                precond = 0.5 * (precond + precond.T)          # symmetrise
                print('Preconditioner matrix')
                print(f"eigvals: {np.linalg.eigvalsh(precond)}")
                print(f'condition no. {condition_number(precond)}')
        else:
            print("incomplete Cholesky broke down (A not numerically PD)- "
                "skipping preconditioned spectrum")

        iterations = res.iterations
        num_iters = iterations[-1]['iter']
        grad = iterations[-1]['grad']
        print(f"min residual value {l2_norm(grad)} found at {res.x} in {num_iters} iterations")
        print(f'||Ax-b||: {l2_norm(A @ res.x - b)}')
        print(f'scipy solve: {scipy.linalg.solve(A, b)}')

        #plot_cg_convergence(A, b, res)                        
        plot_cg_convergence(A, b, res, spectrum='preconditioned', spectrum_style='hist') # density histogram


def run_nonlinear_cg():
    #x0 = [1.2, 1.2]
    x0 = np.array([-1.2, 1.0])

    optimizer_params = {'conv_tol': 1e-6, 'max_iter':200}

    line_search_args = {'a0': 1.0}
    res = minimize(rosenbrock, x0, line_search_method=backtracking, conjugate_method=polak_ribiere, 
                   line_search_args=line_search_args, 
                   **optimizer_params)
    grad = res.iterations[-1]['grad']
    print(f'CG converged: {res.converged} to min f(x) {res.f_min} at {res.x} in {res.terminal} iterations')
    print(f"min residual norm {l2_norm(grad)}")
    plot_nonlinear_cg_convergence(res)

def run_newton_cg():
    from presto.inexact_newton import newton_cg, newton_lanczos
    #x0 = [1.2, 1.2]
    x0 = np.array([-1.2, 1.0])

    optimizer_params = {'conv_tol': 1e-5, 'max_iter':200}
    res = newton_cg.minimize(rosenbrock, x0, 
             line_search=False,
             line_search_method=None,
             preconditioner=None,
             **optimizer_params)
    plot_nonlinear_cg_convergence(res)

def main():

    run_linear_cg()

    plt.show(block=True)

OBJECTIVES = {
}

if __name__ == "__main__":
    main()




