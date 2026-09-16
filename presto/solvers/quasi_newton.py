import numpy as np
from presto.linalg import l2_norm, inverse

machine_eps = np.finfo(np.float32).eps

def secant(func, x_cur, method, x_next=None):
    if x_next is None or np.isclose(x_cur, x_next):
        eps = 1e-4
        x_next = x_cur * (1 + eps)
        x_next += (eps if x_next >= 0 else -eps)
    f_cur = func(x_cur)
    f_next = func(x_next)
    if np.ndim(x_cur) == 0:
        return (f_next - f_cur) / (x_next - x_cur)
    return method(x_cur, f_cur, x_next, f_next)
   
def broyden(x_cur, f_cur, x_next=None, f_next=None, j_cur=None, inv=False, scale=1.0):
    if j_cur is None:
        mult = scale/l2_norm(f_cur) if not inv else l2_norm(f_cur)/scale
        j_cur = mult * np.identity(len(x_cur))  
    if x_next is None:
        return j_cur  
    y = f_next - f_cur
    s = x_next - x_cur 
    if inv:
        a = j_cur @ y
        b = s @ j_cur 
        update = np.outer(s - a, b) / (s @ a)
    else:
        update = np.outer(y - j_cur @ s, s) / (s @ s)
    j_next = j_cur + update
    return j_next

def bfgs(x_cur, g_cur, x_next=None, g_next=None, B_cur=None, inv=False, scale=1.0):
    if B_cur is None:
        #mult = scale/l2_norm(g_cur) if not inv else l2_norm(g_cur)/scale
        mult=scale
        B_cur = mult * np.identity(len(x_cur))  
    if x_next is None:
        return B_cur  
    s = x_next - x_cur 
    y = g_next - g_cur 
    ys = y @ s
    if ys <= 0: #if y.T s <= 1e-12 * l2_norm(s) * l2_norm(y): return B_cur
        raise ValueError("Curvature condition violated - nonpositive curvature encountered. Secant equation only accepts positive/convex Hessian approximations")
    if inv:
        rho = 1 / ys
        I = np.identity(len(s))
        a = rho * np.outer(s, y)
        b = I - a
        return b @ B_cur @ b.T + rho * np.outer(s, s)
    a = B_cur @ s
    B_next = B_cur - np.outer(a, a.T) / (a.T @ s) + np.outer(y, y.T) / ys
    return B_next

def damped_bfgs(x_cur, g_cur, x_next=None, g_next=None, B_cur=None, inv=False, scale=1.0):
    if B_cur is None:
        B_cur = scale * np.identity(len(x_cur))
    if x_next is None:
        return B_cur
    s = x_next - x_cur
    y = g_next - g_cur
    B = inverse(B_cur) if inv else B_cur
    Bs = B @ s
    sBs = s @ Bs
    if sBs <= machine_eps:
        return B_cur
    sy = s @ y
    theta = 1.0 if sy >= 0.2 * sBs else 0.8 * sBs / (sBs - sy)
    r = theta * y + (1 - theta) * Bs
    sr = s @ r
    if sr <= machine_eps:
        return B_cur
    if inv:
        rho = 1 / sr
        I = np.identity(len(s))
        a = rho * np.outer(s, r)
        b = I - a
        return b @ B_cur @ b.T + rho * np.outer(s, s)
    return B_cur - np.outer(Bs, Bs) / sBs + np.outer(r, r) / sr

def symmetric_rank_one(x_cur, g_cur, x_next=None, g_next=None, B_cur=None, inv=False, scale=1.0, den_tol=1e-6):
    if B_cur is None:
        #mult = scale/l2_norm(g_cur) if not inv else l2_norm(g_cur)/scale
        mult=scale
        B_cur = mult * np.identity(len(x_cur))  
    if x_next is None:
        return B_cur   
    s = x_next - x_cur 
    y = g_next - g_cur 
    c = y @ s 
    if inv:
        m = y
        k = s
    else:
        m = s 
        k = y
    a = B_cur @ m
    b = k - a 
    if np.allclose(b, 0.0, atol=den_tol):
        return B_cur
    d = m @ a 
    e = c - d            
    if np.abs(e) < den_tol * l2_norm(b) * l2_norm(m):
        return B_cur
    update = np.outer(b, b) / e   
    B_next = B_cur + update
    return B_next

QUASI_NEWTON_HESSIAN_UPDATES = {
    'bfgs': bfgs,
    'damped bfgs': damped_bfgs,
    'damped_bfgs': damped_bfgs,
    'sr1': symmetric_rank_one,
    'symmetric rank 1': symmetric_rank_one,
    'symmetric rank one': symmetric_rank_one,
    'broyden': broyden
}
