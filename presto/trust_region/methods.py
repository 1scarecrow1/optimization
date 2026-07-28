import numpy as np
from presto.gradients import gradient
import presto.solvers.quasi_newton as qn 
from presto.trust_region.trust_region import * 
from functools import partial  

def sr1_trust_region(func, x, trust_region_method=dogleg,
                    grad=None, hess=None, func_args = None, 
                    solver_args=None, trust_region_args = None,
                    D=None,
                    conv_tol=1e-5, max_iter=200):
    rad_max = np.inf  
    func_args = func_args or {}
    trust_region_args = trust_region_args or {}

    f = partial(func, **func_args)
    g = partial(gradient, **func_args) if not isinstance(gradient, partial) else gradient 
    B = partial(qn.symmetric_rank_one, inv=True, scale=1.0, den_tol=1e-6)

    x_cur = x
    rad_cur = trust_region_args['rad0'] if trust_region_args else 1.0
    f_cur = f(x_cur) 
    g_cur = g(x_cur) 
    B_cur = B(x_cur, g_cur)

    model = partial(quadratic_model, D=D)
    tr_method = partial(**trust_region_args)

    num_iters = 0
    while not (converged(g_cur, conv_tol) or num_iters >= max_iter): 
        p_cur, x_next, rad_cur = solve_sr1(f, model, x_cur, f_cur, g_cur, B_cur, rad_cur, tr_method)
        g_next = g(x_next)
        B_cur = B(x_cur, g_cur, x_next, g_next, B_cur)
        x_cur = x_next 
        g_cur = g_next
        
