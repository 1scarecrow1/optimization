import numpy as np
from presto.test_functions import *
from presto.linalg import *
from presto.line_search.line_search import *
from presto.solvers.solvers import *
from presto.line_search.optimize import *
import matplotlib.pyplot as plt
from presto.utils import timer
from presto.plotting import plot_contour, plot_iterations
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

    func_name = result.func.__name__
    f_min = result.f_min
    x_min = result.x
    iterations = result.iterations
    line_search_method = result.line_search_method if isinstance(result.line_search_method, str) else result.line_search_method.__name__
    direction = result.solver if isinstance(result.solver, str) else result.solver.__name__

    print(f"Minimising {func_name} with {direction} - {line_search_method}")
    print(f"min value: {round(f_min, 8)}, at {x_min}, found in {len(iterations)-1} iterations")

    if display_all:
        display_keys = ['iter', 'step', 'x', 'func', 'grad']
        for iteration in iterations:
            line = " | ".join(
                f"{key}: {iteration[key]}"
                for key in display_keys
            )
            print(line)

    if save_results:
        pass

    #return summary, 

def plot_results(result, func_name=None, save_results=False):
    iterations = result.iterations 
    func = result.func
    func_name = result.func.__name__ if func_name is None else func_name

    iterations = result.iterations
    line_search_method = result.line_search_method if isinstance(result.line_search_method, str) else result.line_search_method.__name__
    direction = result.solver if isinstance(result.solver, str) else result.solver.__name__
    x_vals = np.array([it["x"] for it in iterations])
    func_vals = np.array([it["func"] for it in iterations])
    grad_vals = np.array([it["grad"] for it in iterations])
    search_directions = np.array([it["step"] for it in iterations])  

    plot_contour(func, x_vals, direction, line_search_method, func_name=func_name) 
    plot_iterations(func_vals, grad_vals, search_directions, func_name=func_name, 
                    method=line_search_method, direction=direction) 

    if save_results:
        pass


def parse_x0(s):
    return np.array([float(v) for v in s.split(",")])

def run_optimizer(objective_func, x0, direction, line_search_method, func_name=None, prev_alpha=False, save_results=False, **params):
    res = minimize(objective_func, x0, direction, line_search_method, prev_alpha=prev_alpha, **params)
    summarise_search(res, display_all=True, save_results=save_results) 
    plot_results(res, func_name=func_name, save_results=save_results)

def main():
    """
    TODO: create function for easy comparison of any 2+ methods - direction and step length
    """
    #x0 = np.array([-1.2, 1.0])
    x0 = np.array([1.2, 1.2])

    objective_func = rosenbrock 
    #func_summary = summarise_function(objective_func, x0)
    #print(func_summary)   
    
    objective_func = lambda x: np.sum(10*x**2 - np.sin(x))
    #objective_func = lambda x: l2_norm(np.sin(x))

    # n = 4
    # c = np.arange(n)
    # xx, yy = np.meshgrid(c, c)
    # A = 1 / (xx + yy + 1)
    # b = np.ones(n)
    # x0 = np.zeros(n)
    # phi = lambda x: A @ x - np.sin(x)
    # objective_func = lambda x: 0.5*(phi(x)-b).T @ (phi(x)-b)

    line_search_methods = {
        'backtracking': {'a0': 1.0, 'c1': 1e-4, 'rho': 0.5},
        'wolfe': {'a0': 1.0, 'a_max': 10.0, 'c1':1e-4, 'c2':0.9, 'max_iter':10, 'zoom_iter':10, 'interp_method':"cubic"},
    }

    initial_step = {'a0': 1.0}

    sd_params = {
        'solver': 'gradient_descent', 
        'line_search_method': 'backtracking', 
        'solver_args': {},
        'line_search_args': line_search_methods['backtracking']
    }
    newton_params = {
        'solver': 'newton', 'line_search_method': 'backtracking', 
        'solver_args': {},
        'line_search_args': line_search_methods['backtracking']
    }
    quasi_newton_args = {'update_inv': True}

    bfgs_params = {
        'solver': 'bfgs', 'line_search_method': 'backtracking', 
        'solver_args': quasi_newton_args,
        'line_search_args': line_search_methods['backtracking']
    }

    sr1_params = {
        'solver': 'sr1', 'line_search_method': 'backtracking', 
        'solver_args': quasi_newton_args,
        'line_search_args': line_search_methods['backtracking']
    }
    modified_newton_params = {
        'solver': 'modified_newton', 'line_search_method': 'backtracking', 
        'solver_args': {'mod_method': gauss_newton_approx},
        'line_search_args': line_search_methods['backtracking']
    }
    optimizer_params = {'conv_tol': 1e-5, 'max_iter':200}
    save_results=False


    # res_s = minimize(objective_func, x0, **sd_params,
    #          prev_alpha=False, 
    #          **optimizer_params)    
    # summarise_search(res_s, display_all=False) 
    # plot_results(res_s, save_results=save_results)

    # res_n = minimize(objective_func, x0, **newton_params,                 
    #                 prev_alpha=False, **optimizer_params)    
    # summarise_search(res_n, display_all=False) 
    # plot_results(res_n, save_results=save_results)

    # res_bb = minimize(objective_func, x0, 'bfgs', 'backtracking', 
    #          solver_args=quasi_newton_args,
    #          line_search_args = line_search_methods['backtracking'],
    #          prev_alpha=False, 
    #          **optimizer_params)
    # summarise_search(res_bb, display_all=False) 
    # plot_results(res_bb, save_results=save_results)

    res_bw = minimize(objective_func, x0, symmetric_rank_one, 'wolfe', 
             solver_args=quasi_newton_args,
             line_search_args = line_search_methods['wolfe'],
             prev_alpha=False, 
             **optimizer_params)
    summarise_search(res_bw, display_all=False) 
    plot_results(res_bw, save_results=save_results)

    # res_mn = minimize(objective_func, x0, **modified_newton_params,                 
    #                 **optimizer_params)    
    # summarise_search(res_mn, display_all=False) 
    # plot_results(res_mn, save_results=save_results)

                
    plt.show(block=True)

OBJECTIVES = {
    "rosenbrock": rosenbrock,
}



def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("--objective_func", default="rosenbrock")
    parser.add_argument("--x0", type=parse_x0, default="-1.2,1.0")
    parser.add_argument("--direction", default="newton")
    parser.add_argument("--line_search", default="backtracking")
    parser.add_argument("--prev_alpha", default=False)

    parser.add_argument("--a0", type=float, default=1)
    parser.add_argument("--c", type=float, default=1e-4)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--max_iter", type=int, default=100)

    args = parser.parse_args()

    result = minimize(
        OBJECTIVES[args.objective],
        args.x0,
        DIRECTIONS[args.direction],
        LINE_SEARCH_METHODS[args.line_search],
        a0=args.a0,
        prev_alpha=args.prev_alpha,
        c=args.c,
        rho=args.rho,
        eps=args.eps,
        max_iter=args.max_iter,
    )

    print(result)

if __name__ == "__main__":
    main()




