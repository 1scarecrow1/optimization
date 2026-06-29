import numpy as np
from test_functions import *
from norms import l2_norm
import matplotlib.pyplot as plt
from functools import partial
from utils import timer
from plotting import plot_contour, plot_iterations

plt.ion()

def steepest_descent(func, x, grad=None, *args, **kwargs):
    """
    Need to add general gradient function
    """
    if grad is not None:
        p = - grad
    else:
        p = - func.gradient(x, *args, **kwargs) 
    return p 

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

def newton_direction(func, x, grad=None, hessian=None, *args, **kwargs):
    if grad is not None:
        g = grad 
    else:
        g = func.gradient(x, *args, **kwargs)
    if hessian is not None:
        H = hessian 
    else:
        H = func.hessian(x, *args, **kwargs)
    return - invert_matrix(H) @ g


def quasi_newton(func, x, *args, **kwargs):
    g = func.gradient(x, *args, **kwargs)
    B = bfgs(func, x)
    p = - B @ g
    return p

@timer
def line_search(func, x, search_direction, a0=1, prev_alpha=False, c = 1e-4, rho=0.5, eps=1e-5, max_iter=100, *args, **kwargs):
    x = np.asarray(x, dtype=float)
    f = partial(func, *args, **kwargs)
    g = partial(func.gradient, *args, **kwargs)
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
        alpha, x_next, f_next = backtracking_line_search(func, x_cur, f_cur, g_cur, p_cur, a0, c = c, rho=rho, *args, **kwargs)
        x_cur = x_next 
        f_cur = f_next 
        g_cur = g(x_cur) 
        p_cur = p(x_cur, g_cur)

        iterations.append({'step': alpha, 'x': x_cur, 'func': f_cur, 'grad': g_cur, 'search_direction': p_cur})
        num_iters += 1
  
        if prev_alpha:
            a0 = alpha

    return x_cur, f_cur, iterations

def backtracking_line_search(func, x_cur, f_cur, g_cur, p_cur, a0, c = 1e-4, rho=0.5, *args, **kwargs):
    if a0 < 0:
        raise ValueError("step length must be positive")
    if not (0 < c < 1):
        raise ValueError("sufficient decrease param must be between 0 and 1")
    if not (0 < rho < 1):
        raise ValueError("step length contraction factor must be between 0 and 1")
    
    alpha = a0
    f = partial(func, *args, **kwargs)  
    x_next = x_cur + alpha * p_cur
    f_next = f(x_next) 

    while f_next > f_cur + c * alpha * g_cur @ p_cur:
        alpha *= rho
        x_next = x_cur + alpha * p_cur
        f_next = f(x_next)

    return alpha, x_next, f_next

def check_convergence(gradient, eps=1e-4):
    """
    TODO: Use test_terminal_steepest_descent 
    """      
    return np.allclose(l2_norm(gradient), 0, rtol=eps)

def minimise(func):
    pass

def bfgs(func, x):
    pass

def invert_matrix(A):
    return np.linalg.inv(A)
    


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
    x0 = np.array([-1.2, 1.0])
    #x0 = np.array([1.2, 1.2])
    objective_func = rosenbrock
    summarise_function(objective_func, x0)

    s_min_x, s_min_f, s_iterations = line_search(objective_func, x0, steepest_descent, prev_alpha=False)
    summarise_search(objective_func, steepest_descent, s_min_x, s_min_f, s_iterations, save_results=False)

    n_min_x, n_min_f, n_iterations = line_search(objective_func, x0, newton_direction)
    summarise_search(objective_func, newton_direction, n_min_x, n_min_f, n_iterations, save_results=False)

    plt.show(block=True)

if __name__ == "__main__":
    main()




