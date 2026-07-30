import warnings

import numpy as np
from presto.linalg import l2_norm
from presto.trust_region.trust_region import solve_quadratic


def cg_steihaug(g_cur, B_cur, rad_cur, tol_cur=None, max_iter=100):
    z_cur = np.zeros_like(g_cur)
    r_cur = g_cur
    if tol_cur is None:
        g_cur_norm = l2_norm(g_cur)
        tol_cur = min(0.5, np.sqrt(g_cur_norm)) * g_cur_norm
    if l2_norm(r_cur) < tol_cur:
            return z_cur
    d_cur = -r_cur 
    e = B_cur @ d_cur 
    rr = r_cur @ r_cur
 
    for _ in range(max_iter):
        curvature = d_cur @ e
        if curvature <= 0:
            a = d_cur @ d_cur 
            b = 2*z_cur @ d_cur 
            c = z_cur @ z_cur - rad_cur**2 
            tau, _ = solve_quadratic(a, b, c)
            return z_cur + tau*d_cur 
        
        alpha = rr / curvature
        z_next = z_cur + alpha * d_cur

        if l2_norm(z_next) >= rad_cur:
            a = d_cur @ d_cur 
            b = 2*z_cur @ d_cur 
            c = z_cur @ z_cur - rad_cur**2 
            r, _ = solve_quadratic(a, b, c)
            tau = max(r, 0.0)
            return z_cur + tau*d_cur
          
        r_next = r_cur + alpha * e 
        if l2_norm(r_next) < tol_cur:
            return z_next 
        rr_next = r_next @ r_next 
        beta = rr_next / rr
        d_cur = -r_next + beta*d_cur 
        e = B_cur @ d_cur
        r_cur, z_cur = r_next, z_next
        rr = rr_next

    print(f'failed to converge after {max_iter} iterations')
    warnings.warn(
        f"CG Steihaug failed to converge after {max_iter} iterations"
        f"returning the last direction", RuntimeWarning)
    return z_cur  