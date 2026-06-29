import numpy as np
from presto.test_functions import *
from presto.linalg import *
from presto.line_search.step_length import backtracking
from presto.line_search.direction import gradient_descent, newton, quasi_newton
import matplotlib.pyplot as plt
from functools import partial
from presto.utils import timer
from presto.plotting import plot_contour, plot_iterations

plt.ion()


def test_terminal_steepest_descent(p, grad):
    grad_norm = l2_norm(grad)
    p_norm = l2_norm(p)
    grad_val = - grad / grad_norm
    angle = - grad @ p / (grad_norm * p_norm)
    converges = np.allclose(p, grad_val, rtol=1e-5)

    res = {
        "Steepest descent direction is unit negative gradient vector": converges, 
        "search direction": p,
        "Angle is minimum": angle == 1,
        "Angle": angle
    }
    if not converges:
        res["Search direction off by"] = 100 * l2_norm(p - grad_val) / p_norm

    return res


@timer
def minimize(func, x, search_direction, a0=1, prev_alpha=False, c = 1e-4, rho=0.5, eps=1e-5, max_iter=100, *args, **kwargs):
    x = np.asarray(x, dtype=float)
    f = partial(func, *args, **kwargs)
    g = partial(func.gradient, *args, **kwargs) # if gradient defined explicitly, else general gradient approximation method
    p = partial(search_direction, func, *args, **kwargs)

    x_cur = x
    f_cur = f(x_cur) 
    g_cur = g(x_cur) 
    p_cur = p(x_cur, g_cur)
    alpha = a0

    iterations = [{'step': 0, 'x': x_cur, 'func': f_cur, 'grad': g_cur, 'search_direction': p_cur}]

    converged = partial(check_convergence, eps=eps) 
    num_iters = 0

    while not (converged(g_cur) or num_iters >= max_iter):
        alpha, x_next, f_next = backtracking(func, x_cur, f_cur, g_cur, p_cur, a0, c = c, rho=rho, *args, **kwargs)
        x_cur = x_next 
        f_cur = f_next 
        g_cur = g(x_cur) 
        p_cur = p(x_cur, g_cur)

        iterations.append({'step': alpha, 'x': x_cur, 'func': f_cur, 'grad': g_cur, 'search_direction': p_cur})
        num_iters += 1
  
        if prev_alpha:
            a0 = alpha

    return x_cur, f_cur, iterations



def check_convergence(gradient, eps=1e-4):
    """
    TODO: Use test_terminal_steepest_descent 
    """      
    return np.allclose(l2_norm(gradient), 0, rtol=eps) 


def summarise_function(func, x):
    r = func(x)
    g = func.gradient(x)
    h = func.hessian(x)
    print(f"func value at {x}: {r}")
    print(f"gradient: {g}")
    print(f"Hessian: {h}")
    

def summarise_search(func, line_search_method, min_x, min_f, iterations, save_results=False):
    print(f"{line_search_method.__name__} for {func.__name__}")
    print(f"Min value: {round(min_f, 4)}, at {min_x}, found in {len(iterations)-1} iterations")
    steps = np.array([it["step"] for it in iterations])
    x_vals = np.array([it["x"] for it in iterations])
    func_vals = np.array([it["func"] for it in iterations])
    grad_vals = np.array([it["grad"] for it in iterations])
    search_directions = np.array([it["search_direction"] for it in iterations])

    print(f"steps: {steps}")
    plot_contour(func, x_vals, method=line_search_method)
    plot_iterations(func_vals, grad_vals, search_directions, func=func, method=line_search_method)

    if save_results:
        pass

def main():
    """
    create function for easy comparison of any 2+ methods - direction and step length
    """
    x0 = np.array([-1.2, 1.0])
    #x0 = np.array([1.2, 1.2])
    objective_func = rosenbrock
    summarise_function(objective_func, x0)

    s_min_x, s_min_f, s_iterations = minimize(objective_func, x0, gradient_descent, prev_alpha=False)
    summarise_search(objective_func, gradient_descent, s_min_x, s_min_f, s_iterations, save_results=False)

    n_min_x, n_min_f, n_iterations = minimize(objective_func, x0, newton)
    summarise_search(objective_func, newton, n_min_x, n_min_f, n_iterations, save_results=False)

    plt.show(block=True)

if __name__ == "__main__":
    main()




