from collections import namedtuple
import numpy as np
from functools import partial
import warnings
import scipy   
from presto.gradients import gradient_func, hessian_func
from presto.linalg import l2_norm
from presto.solvers.newton import newton_iteration, roots
from scipy.linalg import cho_factor, cho_solve, solve_triangular

TrustRegionOutput = namedtuple("TrustRegionOutput", "p")

def quadratic_model(fx, gx, Bx, px, D=None):
    if D is not None:
        d = np.diag(D) if np.ndim(D) == 2 else D
        px = d * px
        d_inv = 1/d
        gx = d_inv * gx 
        Bx = d_inv * (d_inv * Bx).T
    return fx + gx.T @ px + 0.5 * px.T @ Bx @ px  

def trust_region_constraint(radius, px, lamb, D=None):
    if D is not None:
        px = D @ px 
    return - 0.5 * lamb * (radius**2 - px.T @ px)
    
def constrained_model(fx, gx, Bx, radius, p, lamb, D=None):
    obj = quadratic_model(fx, gx, Bx, p, D=D)
    constraint = trust_region_constraint(radius, p, lamb, D=D)
    return obj + constraint

def elliptical_model(fx, gx, Bx, px, D):
    d = np.diag(D) if np.ndim(D) == 2 else D
    px = d * px
    d_inv = 1/d
    gx = d_inv * gx 
    Bx = d_inv * (d_inv * Bx).T
    return fx + gx @ px + 0.5 * px @ Bx @ px 

def reduction_ratio(f_cur, f_next, m_next):
    actual = f_cur - f_next
    predicted = f_cur - m_next
    if predicted <= np.finfo(float).eps * max(1.0, abs(f_cur)):
        return np.inf if f_next <= f_cur else 0.0
    return actual / predicted 

def general(func, model, x_cur, f_cur, g_cur, B_cur, rad_cur, 
            method, method_args = None, solver_args = None, 
            ):
    
    solver_args = solver_args or {}
    method_args = method_args or {}
    expand_condition = solver_args.get('expand_condition', None)
    if expand_condition is None:
        expand_condition = lambda p, rad: np.isclose(l2_norm(p), rad)
    eta = solver_args.get('eta', 0.1)
    rad_max = solver_args.get('rad_max', np.inf)
    thresholds = solver_args.get('thresholds', {})
    rho_min = thresholds.get('rho_min', 0.25)
    rho_max = thresholds.get('rho_max', 0.75) 
    contraction = thresholds.get('contraction', rho_min) 
    expansion = thresholds.get('expansion', 2) 

    p = method(g_cur, B_cur, rad_cur, **method_args)
    x_next = x_cur + p
    f_next = func(x_next)
    m_next = model(f_cur, g_cur, B_cur, p)
    rho_cur = reduction_ratio(f_cur, f_next, m_next)   
    print(f'reduction ratio: {rho_cur}')
    if rho_cur > eta:
        x_cur = x_next
        f_cur = f_next
        p_cur = p 
    else:
        p_cur = np.zeros_like(x_cur)

    if rho_cur < rho_min:
        rad_cur = rad_cur * contraction
    if rho_cur > rho_max and expand_condition(p, rad_cur) :
        rad_cur = min(expansion*rad_cur, rad_max)

    return p_cur, x_cur, f_cur, rad_cur

def cauchy_point(g_cur, B_cur, rad_cur, D=None):
    if D is not None:
        return generalised_cauchy_point(g_cur, B_cur, rad_cur, D)
    g_norm = l2_norm(g_cur)
    p_min = - rad_cur * g_cur / g_norm
    curvature = g_cur @ B_cur @ g_cur
    tau = 1 if curvature <= 0 else min(g_norm**3/(rad_cur*curvature), 1) 
    return tau * p_min

def generalised_cauchy_point(g_cur, B_cur, rad_cur, D):
    d = 1 / np.diag(D) if np.ndim(D) == 2 else 1/D
    g_cur = d * g_cur 
    a = d * B_cur
    B_cur = d * a.T
    # D_inv = np.diag(1 / np.diag(D)), B_cur = D_inv @ B_cur @ D_inv  
    g_norm = l2_norm(g_cur)
    p_min = - rad_cur * d * g_cur / g_norm
    curvature = g_cur @ B_cur @ g_cur
    tau = 1 if curvature <= 0 else min(g_norm**3/(rad_cur*curvature), 1) 
    return tau * p_min

def dogleg(g_cur, B_cur, rad_cur, rad_tol=1e-3):
    curvature = g_cur @ B_cur @ g_cur
    if curvature <= 0 or rad_cur < rad_tol:
        return - rad_cur * g_cur / l2_norm(g_cur) # fall back to steepest descent
    try:
        L = cho_factor(B_cur)                    
        p_b = cho_solve(L, -g_cur)
    except np.linalg.LinAlgError:
        return -rad_cur * g_cur / l2_norm(g_cur)  # B not PD -> fall back to steepest descent
    if l2_norm(p_b) <= rad_cur:
        return p_b
    p_u = - (g_cur @ g_cur / curvature) * g_cur
    p_u_norm = l2_norm(p_u)
    if p_u_norm >= rad_cur:
        return rad_cur * p_u/p_u_norm 
    dif = p_b - p_u 
    a = dif @ dif 
    b = 2*dif @ p_u
    c = p_u @ p_u - rad_cur ** 2 
    r, _ = solve_quadratic(a, b, c)      # positive root, c ≤ 0 when ‖p_u‖ ≤ rad
    s = max(r, 0.0)
    p_tau = p_u + s * dif        # r = tau - 1
    return p_tau 

def two_dim_subspace_min(g_cur, B_cur, rad_cur):
    pass 

def trust_region_subproblem(g_cur, B_cur, rad_cur, lamb_tol=1e-8, max_iter=50):
    '''
    Exact trust-region subproblem

        min_p  gᵀp + ½ pᵀBp     s.t.  ‖p‖₂ ≤ Δ

    Safeguarded Newton iteration on the secular equation (Moré & Sorensen 1983)

        φ(λ) = 1/Δ - 1/‖p(λ)‖ = 0,   (B + λI)p(λ) = -g,   λ ≥ max(0, -λ_min(B))

    φ is nearly linear in λ, so Newton needs a handful of iterations, each one
    Cholesky. Definiteness is decided by whether that Cholesky succeeds and λ is
    bracketed by O(n²) norm bounds, so no eigendecomposition is formed on the
    ordinary paths - only in the hard case, which is rare.

    Hard case - ‖p(λ)‖ < Δ with λ pinned at -λ_min(B): g has (numerically) no component along
    the smallest eigenvector z, so no λ ever puts p on the boundary. The solution
    is p* = p + τz with ‖p + τz‖ = Δ.

    This is the one branch where an eigendecomposition earns its cost: it is rare,
    and it is the only place the eigenvector itself - not just λ_min - is needed.
    '''
    g_cur = np.asarray(g_cur, dtype=float)
    B_cur = np.asarray(B_cur, dtype=float)
    n = B_cur.shape[0]
    try:
        p = cho_solve(cho_factor(B_cur, lower=True), -g_cur)
        if l2_norm(p) <= rad_cur:
            return p               # λ* = 0, constraint inactive
    except np.linalg.LinAlgError:
        pass                       # B not PD -> λ* > 0, boundary

    g_norm = l2_norm(g_cur)
    B_norm = np.abs(B_cur).sum(axis=1).max()
    B_diag = np.diagonal(B_cur).copy()
    a = g_norm/rad_cur
    lamb_lo = max(0.0, -B_diag.min(), a - B_norm) 
    lamb_hi = max(lamb_lo, a + B_norm)
    lamb = min(max(lamb_lo, a), lamb_hi)

    W = np.empty_like(B_cur)
    diag = np.diag_indices(n)
    p = np.zeros_like(g_cur)
    p_norm = 0.0 

    for _ in range(max_iter):
        np.copyto(W, B_cur)
        W[diag] = B_diag + lamb # or W.flat[::n+1] = B_diag + lamb
        try:
            L, lower = cho_factor(W, lower=True)
        except np.linalg.LinAlgError:
            # λ too small for B + λI to be PD: raise the lower bound and retry.
            lamb_lo = lamb
            lamb = max(np.sqrt(lamb_lo * lamb_hi),
                       lamb_lo + 0.01 * (lamb_hi - lamb_lo))
            continue

        p = cho_solve((L, lower), -g_cur)
        p_norm = np.sqrt(p @ p)

        if abs(p_norm - rad_cur) <= lamb_tol * rad_cur:
            return p

        if p_norm > rad_cur:
            lamb_lo = lamb                            # under-damped
        else:
            lamb_hi = lamb                            # over-damped
            # hard case: bracket collapsed but ‖p‖ is still short of Δ
            if lamb_hi - lamb_lo <= lamb_tol * max(1.0, lamb_hi):
                _, V = np.linalg.eigh(B_cur)
                z = V[:, 0]                                       # unit eigenvector of λ_min
                # ‖p + τz‖² = Δ²  ->  τ² + 2(pᵀz)τ + (‖p‖² − Δ²) = 0
                # Either root is optimal: m(p+τz) − m(p) = −(λ/2)(2τ pᵀz + τ²‖z‖²)
                #                                        = −(λ/2)(Δ² − ‖p‖²),  the same for both.
                tau, _ = solve_quadratic(1.0, 2.0 * (p @ z), p_norm**2 - rad_cur**2)
                return p + tau * z

        # Newton on φ. q = L⁻¹p gives φ'(λ) = −‖q‖²/‖p‖³, and the step collapses to
        q = solve_triangular(L, p, lower=True)        # reads the lower triangle only
        lamb_next = lamb + (p_norm / np.sqrt(q @ q))**2 * (p_norm - rad_cur) / rad_cur

        # safeguard: bisect whenever Newton leaves the bracket
        if not (lamb_lo < lamb_next < lamb_hi):
            lamb_next = 0.5 * (lamb_lo + lamb_hi)
        lamb = lamb_next

    warnings.warn(
        f"trust_region_subproblem: no boundary solution in {max_iter} iterations; "
        f"returning the last step scaled to the boundary", RuntimeWarning)
    return p if p_norm == 0.0 else rad_cur * p / p_norm

def trust_region_subproblem2(g_cur, B_cur, rad_cur, lamb0=1.0, lamb_tol=1e-4, max_iter=200):
    eig_min = np.linalg.eigvalsh(B_cur)[0]
    try:
        p = cho_solve(cho_factor(B_cur, lower=True), -g_cur)
        if l2_norm(p) <= rad_cur:
            return p               # λ* = 0, constraint inactive
    except np.linalg.LinAlgError:
        pass                       # B not PD -> λ* > 0, boundary

    eps = np.finfo(np.float32).eps * max(1.0, abs(eig_min)) # scale by min eigenvalue
    e = max(0.0, -eig_min) + eps 
    lamb = max(lamb0, e) 
    I = np.identity(B_cur.shape[0])
    for _ in range(max_iter):
        L, lower = cho_factor(B_cur + lamb*I, lower=True)
        p = cho_solve((L, lower), -g_cur)
        p_norm = l2_norm(p)
        if np.isclose(p_norm, rad_cur, rtol=lamb_tol):
            return p
        q = scipy.linalg.solve_triangular(L, p, lower=True)
        q_norm = l2_norm(q)
        lamb += (p_norm/q_norm)**2 * (p_norm - rad_cur) / rad_cur
        lamb = max(lamb, e)
    print('failed to converge to valid trust region boundary solution')
    return rad_cur * p/p_norm  

def solve_tau():
    dif = p_b - p_u 
    a = dif @ dif 
    b = 2*dif @ p_u
    c = p_u @ p_u - rad_cur ** 2 
    r1, r2 = solve_quadratic(a, b, c)
    return r

def check_convergence(g_cur, conv_tol=1e-4):   
    return l2_norm(g_cur) <= conv_tol 

def newton_root(rad):
    pass

def solve_quadratic(a, b, c):
    '''
    Solve ax^2 + bx + c = 0
    '''
    disc = np.sqrt(b*b - 4*a*c)
    if disc < 0:
        raise ValueError(f"no real root: discriminant {disc}")
    q = -0.5 * (b + np.copysign(np.sqrt(disc), b))
    r1, r2 = q / a, c / q
    return max(r1, r2), min(r1, r2)

TRUST_REGION_METHODS = {
    'dogleg': dogleg,
    'subproblem': trust_region_subproblem,
    'cauchy_point': cauchy_point
    }





























































































































































































































































































































































































































































































































































































































































































































































































































