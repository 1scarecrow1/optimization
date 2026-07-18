import numpy as np
from functools import partial   
from presto.gradients import gradient_func, hessian_func
from presto.linalg import backward_substitution, forward_substitution, l2_norm, solve_cholesky

def quadratic_model(f, x, g, B, p, D=None):
    # f = partial(func, **func_args)
    # gradient = gradient_func(func, grad, **func_args)    
    # g = partial(gradient, **func_args) if not isinstance(gradient, partial) else gradient 
    # hessian = hessian_func(func, hess, **func_args)    
    # B = partial(hessian, **func_args) if not isinstance(hessian, partial) else hessian 
    if D is not None:
        return elliptical_model(f, x, g, B, p, D)
    # f(x) + g(x).T @ p + 0.5 * p.T @ B @ p - 0.5 * lamb * (radius**2 - p.T @ p) 
    return f + g.T @ p + 0.5 * p.T @ B @ p  

def trust_region_constraint(radius, p, lamb, D=None):
    if D is not None:
        p = D @ p 
    return - 0.5 * lamb * (radius**2 - p.T @ p)
    
def constrained_model(f, x, g, B, radius, p, lamb, D=None):
    return f + g.T @ p + 0.5 * p.T @ B @ p - 0.5 * lamb * (radius**2 - p.T @ p) 

def elliptical_model(f, x, g, B, p, D):
    p_e = D @ p
    D_inv = np.linalg.solve(D, np.identity(len(D)))
    return f(x) + g(x).T @ D_inv @ p_e + 0.5 * p_e.T @ D_inv @ B @ D_inv @ p_e 

def model_reduction_ratio(f, x, p, model):
    actual = f(x) - f(x + p)
    predicted = model(0) - model(p)
    if np.isclose(predicted, 0):
        return 0
    return actual / predicted 

def reduction_ratio(f_cur, f_next, m_next):
    actual = f_cur - f_next
    predicted = f_cur - m_next
    if np.isclose(predicted, 0):
        return 0.0
    return actual / predicted 

def general(func, x_cur, f_cur, g_cur, B_cur, rad_cur, rad_max, lamb0, eta, grad=None, hess=None, D=None, **func_args):
    f = partial(func, **func_args)
    model = partial(quadratic_model, f_cur, x_cur, g_cur, B_cur, D=D)

    p, _ = trust_region_subproblem(g_cur, B_cur, lamb0, rad_cur)
    f_next = f(x_cur + p)
    m_next = model(p)
    rho_cur = reduction_ratio(f_cur, f_next, m_next)

    if rho_cur < 1/4:
        rad_next = rad_cur / 4
    else:
        if rho_cur > 3/4 and np.isclose(l2_norm(p), rad_cur):
            rad_next = min(2*rad_cur, rad_max)
        else:
            rad_next = rad_cur 
    if rho_cur > eta:
        x_next = x_cur + p
    else:
        x_next = x_cur

    return p, rad_next, x_next, f(x_next)

def cauchy_point(g_cur, B_cur, rad_cur, D=None):
    if D is not None:
        return elliptical_cauchy_point(g_cur, B_cur, rad_cur, D)
    g_norm = l2_norm(g_cur)
    p_min = - rad_cur * g_cur / g_norm
    tau = 1 if g_cur @ B_cur @ g_cur <= 0 else min(g_norm**3/(rad_cur*g_cur @ B_cur @ g_cur), 1) # handle g, B close to 0
    return tau * p_min

def elliptical_cauchy_point(g_cur, B_cur, rad_cur, D):
    D_inv = np.linalg.solve(D, np.identity(len(D)))
    g_cur = D_inv @ g_cur
    B_cur = D_inv @ B_cur @ D_inv  
    g_norm = l2_norm(g_cur)
    p_min = - rad_cur * D_inv @ g_cur / g_norm
    tau = 1 if g_cur @ B_cur @ g_cur <= 0 else min(g_norm**3/(rad_cur*g_cur @ B_cur @ g_cur), 1) # handle g, B close to 0
    return tau * p_min

def dogleg(g_cur, B_cur, rad_cur, tau):
    p_b = - np.linalg.solve(B_cur, g_cur)
    p_u = - (g_cur @ g_cur / (g_cur @ B_cur @ g_cur)) * g_cur
    p_tau = tau * p_u if 0 <= tau <= 1 else p_u + (tau - 1)*(p_b - p_u) # trajectory
    if l2_norm(p_b) <= rad_cur:
        return p_b 
    dif = p_b - p_u 
    a = dif @ dif 
    b = 2*dif @ p_u
    c = p_u @ p_u - rad_cur ** 2 
    p1, p2 = quadratic_roots(a, b, c)
    if p1 >= 0 and p1 >= p2:
        return p1 
    elif p2 >= 0 and p2 > p1:
        return p2 
    return np.zeros(len(p_b))

def trust_region_subproblem(g_x, B_x, lamb0, rad):
    lamb = lamb0
    L = np.linalg.cholesky(B_x + lamb*np.identity(len(B_x)))
    q = forward_substitution(L, -g_x)
    p = backward_substitution(L.T, q)
    p_norm = l2_norm(p)
    lamb_next = lamb + (p_norm/l2_norm(q))**2 * (p_norm - rad) / rad

    while not np.isclose(lamb_next, lamb):
        L = np.linalg.cholesky(B_x + lamb*np.identity(len(B_x)))
        q = forward_substitution(L, -g_x)
        p = backward_substitution(L.T, q)
        p_norm = l2_norm(p)
        lamb_next = lamb + (p_norm/l2_norm(q))**2 * (p_norm - rad) / rad

    return p, lamb  

def converged(grad, conv_tol=1e-4):
    return l2_norm(grad) < conv_tol

def newton_root(rad):
    pass

def quadratic_roots(a, b, c):
    '''
    Solve ax^2 + bx + c = 0
    '''
    d = np.sqrt(b**2 - 4*a*c) / (2 * a)
    e = - b/(2*a)
    x1 = e + d  
    x2 = e - d 
    return x1, x2


TRUST_REGION_METHODS = {}





























































































































































































































































































































































































































































































































































































































































































































































































































