import numpy as np
from presto.linalg import l2_norm

def linear_cg(A, b, x, preconditioning=False, preconditioner=None):

    if preconditioning:
        return linear_preconditioned_cg(A, b, x, preconditioner)
    
    x_cur = x
    r_cur = A @ x_cur - b 
    p_cur = -r_cur 

    while not converged(r_cur):
        c = A @ p_cur 
        d = p_cur @ c

        alpha = r_cur @ r_cur / d
        x_cur = x_cur + alpha * p_cur
        r_next = r_cur + alpha * c  
        beta = fletcher_reeves(r_cur, r_next)
        p_cur = -r_next + beta * p_cur 
        r_cur = r_next

    return x_cur  

def linear_preconditioned_cg(A, b, x, preconditioner):
    '''
    Find good conditioning matrices
    '''
    x_cur = x
    r_cur = A @ x_cur - b 
    lu = preconditioner(A) 
    y_cur = lu.solve(r_cur)
    p_cur = - y_cur

    while not converged(r_cur):
        c = A @ p_cur 
        d = p_cur @ c
        e = r_cur @ y_cur 

        alpha = e / d
        x_cur = x_cur + alpha * p_cur
        r_next = r_cur + alpha * c  
        y_next = lu.solve(r_next)
        beta = r_next @ y_next / e
        p_cur = -y_next + beta * p_cur 
        r_cur = r_next
        y_cur = y_next

    return x_cur  

def linear_cg_iteration(A, x_cur, r_cur, p_cur, rr=None):
    c = A @ p_cur 
    d = p_cur @ c
    if rr is None:
        rr = r_cur @ r_cur 
    alpha = r_cur @ r_cur / d
    x_cur = x_cur + alpha * p_cur
    r_next = r_cur + alpha * c  
    rr_next = r_next @ r_next 
    beta = rr_next / rr
    p_cur = -r_next + beta * p_cur 
    r_cur = r_next

    return x_cur, r_cur, p_cur, rr_next

def linear_preconditioned_cg_iteration(A, x_cur, r_cur, p_cur, y_cur, lu, ry=None):
    '''
    Find good conditioning matrices
    '''
    c = A @ p_cur 
    d = p_cur @ c
    if ry is None:
        ry = r_cur @ y_cur

    alpha = ry / d
    x_cur = x_cur + alpha * p_cur
    r_next = r_cur + alpha * c  
    y_next = lu.solve(r_next)
    ry_next = r_next @ y_next 
    beta = r_next @ y_next / ry
    p_cur = -y_next + beta * p_cur 
    r_cur, y_cur = r_next, y_next

    return x_cur, r_cur, p_cur, ry_next

def nonlinear_cg(func, x, grad, line_search_method, beta):
    '''
    TODO: Add restart and check for p conjugate
    '''
    x_cur = x
    f_cur = func(x_cur)
    g_cur = grad(x_cur)
    p_cur = - g_cur

    while not converged(g_cur):
        alpha = line_search_method(func, x_cur, f_cur, g_cur, p_cur) 
        x_cur = x_cur + alpha * p_cur 
        g_next = grad(x_cur)
        b = beta(g_cur, g_next, p_cur)
        g_cur = g_next
        f_cur = func(x_cur)
        p_cur = - g_cur + b * p_cur 

    return x_cur   

def fletcher_reeves(g_cur, g_next, p_cur=None):
    return g_next @ g_next / (g_cur @ g_cur)

def polak_ribiere(g_cur, g_next, p_cur=None, pos_only=False):
    beta = g_next @ (g_next - g_cur) / (g_cur @ g_cur)
    if pos_only:
        return max(beta, 0)
    return beta

def hestenes_stiefel(g_cur, g_next, p_cur):
    a = g_next - g_cur
    return g_next @ a / (a @ p_cur)

def dai_yuan(g_cur, g_next, p_cur):
    a = g_next @ g_next 
    b = (g_next - g_cur)
    return a / (b @ p_cur)

def hager_zhang(g_cur, g_next, p_cur):
    a = g_next - g_cur    
    b = a @ p_cur
    c = a - 2*p_cur * (a @ a) / b
    return c @ g_next / b

def pos_beta(beta):
    return max(0, beta)

def converged(x, conv_tol=1e-4):
    return l2_norm(x) < conv_tol

CG_METHODS = {
    'cg': linear_cg,
    'preconditioned cg': linear_preconditioned_cg
}

NONLINEAR_CG_BETAS = {
    'fletcher reeves': fletcher_reeves,
    'fletcher_reeves': fletcher_reeves,
    'FR': fletcher_reeves,
    'polak_ribiere': polak_ribiere,
    'polak ribiere': polak_ribiere,
    'PR': polak_ribiere,
    'hestenes_stiefel': hestenes_stiefel,
    'hestenes stiefel': hestenes_stiefel,
    'HS': hestenes_stiefel,
    'dai_yuan': dai_yuan,
    'dai yuan': dai_yuan,
    'DY': dai_yuan,
    'hager_zhang': hager_zhang,
    'hager zhang': hager_zhang,
    'HZ': hager_zhang
}