import numpy as np
from presto.test_functions import *
from presto.linalg import *
from presto.trust_region.trust_region import *
import presto.solvers.solvers as solver
import presto.solvers.quasi_newton as qn
import presto.solvers.newton as nu
from presto.trust_region.optimize import *
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
    trust_region_method = result.trust_region_method \
        if isinstance(result.trust_region_method, str) else result.trust_region_method.__name__
    direction = result.solver if isinstance(result.solver, str) else result.solver.__name__
    print(f"Minimising {func_name} with {direction} - {trust_region_method}")
    print(f"min value: {round(f_min, 8)}, at {x_min}, found in {len(iterations)-1} iterations")

    if display_all:
        display_keys = ['iter', 'step', 'size', 'x', 'func', 'grad']
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
    method = result.trust_region_method if isinstance(result.trust_region_method, str) else result.trust_region_method.__name__
    #direction = result.search_direction if isinstance(result.search_direction, str) else result.search_direction.__name__
    x_vals = np.array([it["x"] for it in iterations])
    func_vals = np.array([it["func"] for it in iterations])
    grad_vals = np.array([it["grad"] for it in iterations])
    search_directions = np.array([it["step"] for it in iterations])  

    plot_contour(func, x_vals, '', method, func_name=func_name) 
    plot_iterations(func_vals, grad_vals, search_directions, func_name=func_name, 
                    method=method) 

    if save_results:
        pass


def parse_x0(s):
    return np.array([float(v) for v in s.split(",")])

def run_optimizer(objective_func, x0, direction, line_search_method, func_name=None, prev_alpha=False, save_results=False, **params):
    res = minimize(objective_func, x0, direction, line_search_method, prev_alpha=prev_alpha, **params)
    summarise_search(res, display_all=True, save_results=save_results) 
    plot_results(res, func_name=func_name, save_results=save_results)

def run_newton():
    pass 

def run_sr1():
    pass 

def main():
    """
    TODO: create function for easy comparison of any 2+ methods - direction and step length
    """
    #x0 = np.array([-1.2, 1.0])
    x0 = np.array([1.2, 1.2])

    objective_func = rosenbrock    
    func_summary = summarise_function(objective_func, x0)
    print(func_summary)

    # trust_region_methods = {
    #     'general': {'a0': 1.0, 'c': 1e-4, 'rho': 0.5},
    # }


    # bfgs_params = {
    #     'search_direction': 'bfgs', 'line_search_method': 'backtracking', 
    #     'direction_args': quasi_newton_args,
    #     'line_search_args': line_search_methods['backtracking']
    # }

    optimizer_params = {'conv_tol': 1e-3, 'max_iter':500}
    save_results=False
    # trust_region_subproblem
    # res_n = minimize(objective_func, x0, nu.newton, trust_region_method=trust_region_subproblem,                 
    #                 **optimizer_params)    
    # summarise_search(res_n, display_all=False) 
    # plot_results(res_n, save_results=save_results)

    res_bb = minimize(objective_func, x0, qn.bfgs, trust_region_subproblem, 
             **optimizer_params)
    summarise_search(res_bb, display_all=False) 
    plot_results(res_bb, save_results=save_results)
    
    # dogleg only valid when B is positive definite, which SR1 does not guarantee
    # sr1_params = {'expand_condition': lambda p, rad: l2_norm(p) <= 0.8*rad,
    #               'eta': 0.01, 'thresholds': {'rho_min': 0.1, 'rho_max': 0.75, 'contraction': 0.5, 'expansion': 2}}
    # res_sr1 = minimize(objective_func, x0, qn.symmetric_rank_one, dogleg, 
    #          solver_args=sr1_params,
    #          #trust_region_args= line_search_methods['backtracking'],
    #          **optimizer_params)
    # summarise_search(res_sr1, display_all=False) 
    # plot_results(res_sr1, save_results=save_results)

    plt.show(block=True)

OBJECTIVES = {
    "rosenbrock": rosenbrock,
}



if __name__ == "__main__":
    main()




