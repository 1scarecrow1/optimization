from functools import partial
import logging
import warnings
import numpy as np
from presto.conjugate_gradient.conjugate_gradient import projected_cg
from presto.gradients import gradient_func, hessian_func
from presto.linalg import is_PD, is_PSD, is_symmetric, l2_norm
from presto.minimize_res import MinimizeResult
from presto.quadratic_programming.active_set import active_set, feasible_point
from presto.quadratic_programming.interior_point import interior_point
from presto.quadratic_programming.lagrangian import augmented_lagrangian
from presto.quadratic_programming.sequential_qp import line_search_sqp, trust_region_sqp
from presto.utils import timer

logger = logging.getLogger(__name__)

@timer
def minimize(func, x, method='line search', constraints=None,
             equalities=None, inequalities=None, grad=None, jac=None,
             equality_jac=None, inequality_jac=None, hess=None,
             bounds=None, lambdas=None, func_args=None, method_args=None,
             conv_tol=1e-6, max_iter=100):
    
    func_args = func_args or {}
    method_args = method_args or {}

    f = partial(func, **func_args)
    gradient = gradient_func(func, grad, **func_args)
    g = partial(gradient, **func_args) if not isinstance(gradient, partial) else gradient
    x_cur = np.asarray(x, dtype=float)

    if x_cur.ndim != 1:
        raise ValueError("x must have shape (n,)")
    if np.ndim(f(x_cur)) != 0:
        raise ValueError("quadratic_programming.minimize expects a scalar objective")
    if np.asarray(g(x_cur)).shape != x_cur.shape:
        raise ValueError(f"gradient must have shape {x_cur.shape}")
    
    lagrangian_hessian = None
    if hess is not None:
        hessian = hessian_func(func, grad, hess, **func_args)
        h = partial(hessian, **func_args) if not isinstance(hessian, partial) else hessian
        if np.asarray(h(x_cur)).shape != (x_cur.size, x_cur.size):
            raise ValueError(f"hessian must have shape {(x_cur.size, x_cur.size)}")
        lagrangian_hessian = lambda z, multipliers: np.asarray(h(z), dtype=float)

    if 'B0' in method_args:
        B0 = np.asarray(method_args['B0'], dtype=float)
        if B0.shape != (x_cur.size, x_cur.size) or not is_symmetric(B0):
            raise ValueError(f"B0 must be symmetric with shape {(x_cur.size, x_cur.size)}")
        method_args = {**method_args, 'B0': B0}

    if method == 'augmented lagrangian':
        if constraints is None:
            raise ValueError("augmented Lagrangian requires equality constraints")
        
        constraint_jacobian = gradient_func(constraints, jac)
        c = lambda z: np.atleast_1d(np.asarray(constraints(z), dtype=float))
        A = lambda z: np.atleast_2d(np.asarray(constraint_jacobian(z), dtype=float))

        c0 = c(x_cur)
        lambda0 = np.zeros(c0.size) if lambdas is None else np.asarray(lambdas, dtype=float)

        bounds = [(None, None)] * x_cur.size if bounds is None else bounds

        if A(x_cur).shape != (c0.size, x_cur.size):
            raise ValueError(f"constraint Jacobian must have shape {(c0.size, x_cur.size)}")
        if lambda0.shape != c0.shape:
            raise ValueError(f"lambdas must have shape {c0.shape}")
        if len(bounds) != x_cur.size or any(len(pair) != 2 for pair in bounds):
            raise ValueError(f"bounds must contain {x_cur.size} (lower, upper) pairs")
        if any(pair[0] is not None and pair[1] is not None and pair[0] > pair[1]
               for pair in bounds):
            raise ValueError("bound lower limits must not exceed upper limits")
        
        lower = np.array([-np.inf if pair[0] is None else pair[0] for pair in bounds])
        upper = np.array([np.inf if pair[1] is None else pair[1] for pair in bounds])

        if np.any(x_cur < lower) or np.any(x_cur > upper):
            raise ValueError("initial point must satisfy the bounds")
        
        x_opt = augmented_lagrangian(f, x_cur, c, g, A, bounds, lambda0,
                                     eta_tol=conv_tol, omega_tol=conv_tol,
                                     max_iter=max_iter, **method_args)
        
        equalities, equality_jac = c, A
        inequalities = lambda z: np.empty(0)
        inequality_jac = lambda z: np.empty((0, z.size))

    elif method == 'line search':
        equalities = (lambda z: np.empty(0)) if equalities is None else equalities
        inequalities = (lambda z: np.empty(0)) if inequalities is None else inequalities

        equality_gradient = gradient_func(equalities, equality_jac)
        inequality_gradient = gradient_func(inequalities, inequality_jac)

        equality_func = equalities
        inequality_func = inequalities

        c_eq = lambda z: np.atleast_1d(np.asarray(equality_func(z), dtype=float))
        c_ineq = lambda z: np.atleast_1d(np.asarray(inequality_func(z), dtype=float))
        A_eq = lambda z: np.asarray(equality_gradient(z), dtype=float).reshape((-1, z.size))
        A_ineq = lambda z: np.asarray(inequality_gradient(z), dtype=float).reshape((-1, z.size))
        ceq0 = c_eq(x_cur)
        cineq0 = c_ineq(x_cur)

        if A_eq(x_cur).shape != (ceq0.size, x_cur.size):
            raise ValueError(f"equality Jacobian must have shape {(ceq0.size, x_cur.size)}")
        if A_ineq(x_cur).shape != (cineq0.size, x_cur.size):
            raise ValueError(f"inequality Jacobian must have shape {(cineq0.size, x_cur.size)}")
        if lambdas is not None:
            lambdas = np.asarray(lambdas, dtype=float)
            if lambdas.shape != (ceq0.size + cineq0.size,):
                raise ValueError(f"lambdas must have shape {(ceq0.size + cineq0.size,)}")
            
        x_opt = line_search_sqp(f, x_cur, c_eq, c_ineq, g, A_eq, A_ineq,
                                hess=lagrangian_hessian, lambdas=lambdas, conv_tol=conv_tol,
                                max_iter=max_iter, **method_args)
        
        equalities, equality_jac = c_eq, A_eq
        inequalities, inequality_jac = c_ineq, A_ineq

    elif method == 'trust region':
        if constraints is None:
            raise ValueError("trust-region SQP requires equality constraints")
        
        constraint_jacobian = gradient_func(constraints, jac)
        c = lambda z: np.atleast_1d(np.asarray(constraints(z), dtype=float))
        A = lambda z: np.atleast_2d(np.asarray(constraint_jacobian(z), dtype=float))
        c0 = c(x_cur)

        if A(x_cur).shape != (c0.size, x_cur.size):
            raise ValueError(f"constraint Jacobian must have shape {(c0.size, x_cur.size)}")
        
        if lambdas is not None:
            lambdas = np.asarray(lambdas, dtype=float)
            if lambdas.shape != c0.shape:
                raise ValueError(f"lambdas must have shape {c0.shape}")
            
        x_opt = trust_region_sqp(f, x_cur, c, g, A, hess=lagrangian_hessian,
                                 lambdas=lambdas, conv_tol=conv_tol,
                                 max_iter=max_iter, **method_args)
        
        equalities, equality_jac = c, A
        inequalities = lambda z: np.empty(0)
        inequality_jac = lambda z: np.empty((0, z.size))
    else:
        raise NotImplementedError(f"sequential quadratic programming method {method!r} is not implemented")

    ceq = equalities(x_opt)
    cineq = inequalities(x_opt)
    Aeq = equality_jac(x_opt)
    Aineq = inequality_jac(x_opt)

    active = cineq <= np.sqrt(conv_tol)
    A_work = np.vstack((Aeq, Aineq[active]))
    lambda_work = (np.linalg.lstsq(A_work.T, g(x_opt), rcond=None)[0]
                   if A_work.shape[0] else np.empty(0))
    
    stationarity = g(x_opt) - A_work.T @ lambda_work
    converged_flag = (l2_norm(stationarity) <= np.sqrt(conv_tol) and
                      l2_norm(ceq) <= np.sqrt(conv_tol) and
                      np.all(cineq >= -np.sqrt(conv_tol)))
    
    iterations = [
        {'iter': 0, 'x': x_cur, 'func': f(x_cur), 'grad': g(x_cur)},
        {'iter': 1, 'x': x_opt, 'func': f(x_opt), 'grad': stationarity}
    ]
    if converged_flag:
        logger.info(f'{method} SQP converged')
    else:
        warnings.warn(f'{method} SQP did not converge in {max_iter} iterations', RuntimeWarning)
    return MinimizeResult(
        x=x_opt,
        f_min=f(x_opt),
        iterations=iterations,
        func=func,
        solver='SQP',
        method=method,
        converged=converged_flag,
        extras={'constraint': np.concatenate((ceq, cineq)),
                'lambdas': lambda_work}
    )


@timer
def minimize_quadratic(G, c, x=None, method='active set',
                       A_eq=None, b_eq=None, A_ineq=None, b_ineq=None,
                       y=None, lambdas=None, method_args=None,
                       conv_tol=1e-8, max_iter=100):
    method_args = method_args or {}
    G = np.asarray(G, dtype=float)
    c = np.asarray(c, dtype=float)

    if c.ndim != 1:
        raise ValueError("c must have shape (n,)")
    n = c.size
    if G.shape != (n, n):
        raise ValueError(f"G must have shape {(n, n)}, got {G.shape}")
    if not is_symmetric(G):
        raise ValueError("G must be symmetric")
    if not is_PSD(G):
        raise ValueError("G must be positive semidefinite")
    
    A_eq = np.empty((0, n)) if A_eq is None else np.asarray(A_eq, dtype=float).reshape((-1, n))
    A_ineq = np.empty((0, n)) if A_ineq is None else np.asarray(A_ineq, dtype=float).reshape((-1, n))
    b_eq = np.empty(0) if b_eq is None else np.asarray(b_eq, dtype=float)
    b_ineq = np.empty(0) if b_ineq is None else np.asarray(b_ineq, dtype=float)

    if b_eq.shape != (A_eq.shape[0],):
        raise ValueError(f"b_eq must have shape ({A_eq.shape[0]},), got {b_eq.shape}")
    if b_ineq.shape != (A_ineq.shape[0],):
        raise ValueError(f"b_ineq must have shape ({A_ineq.shape[0]},), got {b_ineq.shape}")
    if x is None:
        x_cur = (np.zeros(n) if method == 'interior point' else
                 feasible_point(A_eq, b_eq, A_ineq, b_ineq, np.zeros(n),
                                conv_tol=conv_tol))
    else:
        x_cur = np.asarray(x, dtype=float)
    if x_cur.shape != (n,):
        raise ValueError(f"x must have shape ({n},), got {x_cur.shape}")
    if method != 'interior point' and ((A_eq.size and not np.allclose(A_eq @ x_cur, b_eq, atol=conv_tol)) or
                                       (A_ineq.size and np.any(A_ineq @ x_cur < b_ineq - conv_tol))):
        raise ValueError("initial point must be feasible")
    
    f = lambda z: 0.5 * z @ G @ z + c @ z
    x0 = x_cur.copy()

    if method == 'projected cg':
        if A_ineq.size:
            raise ValueError("projected CG accepts equality constraints only")
        if np.linalg.matrix_rank(A_eq, tol=conv_tol) < A_eq.shape[0]:
            raise ValueError("A_eq must have full row rank")
        if 'H' in method_args:
            H = np.asarray(method_args['H'], dtype=float)
            if H.shape != G.shape or not is_PD(H):
                raise ValueError(f"H must be positive definite with shape {G.shape}")
            method_args = {**method_args, 'H': H}
        x_opt = projected_cg(G, c, A_eq, b_eq, x_cur,
                             conv_tol=conv_tol, max_iter=max_iter, **method_args)
    elif method == 'active set':
        x_opt = active_set(G, c, A_eq, b_eq, A_ineq, b_ineq, x_cur,
                           conv_tol=conv_tol, max_iter=max_iter, **method_args)
    elif method == 'interior point':
        if A_eq.size or not A_ineq.size:
            raise ValueError("interior point accepts inequality constraints only")
        slack = np.maximum(A_ineq @ x_cur - b_ineq, 1.0) if y is None else np.asarray(y, dtype=float)
        lambda0 = np.ones(A_ineq.shape[0]) if lambdas is None else np.asarray(lambdas, dtype=float)
        if slack.shape != b_ineq.shape or lambda0.shape != b_ineq.shape:
            raise ValueError(f"y and lambdas must have shape {b_ineq.shape}")
        if np.any(slack <= 0) or np.any(lambda0 <= 0):
            raise ValueError("y and lambdas must be strictly positive")
        x_opt = interior_point(G, c, A_ineq, b_ineq, x_cur, slack, lambda0,
                               conv_tol=conv_tol, max_iter=max_iter, **method_args)
    else:
        raise NotImplementedError(f"quadratic programming method {method!r} is not implemented")

    active = np.isclose(A_ineq @ x_opt, b_ineq, atol=np.sqrt(conv_tol))
    A_work = np.vstack((A_eq, A_ineq[active]))
    lambda_work = (np.linalg.lstsq(A_work.T, G @ x_opt + c, rcond=None)[0]
                   if A_work.shape[0] else np.empty(0))
    
    multipliers = np.zeros(A_eq.shape[0] + A_ineq.shape[0])
    multipliers[:A_eq.shape[0]] = lambda_work[:A_eq.shape[0]]
    multipliers[A_eq.shape[0]:][active] = lambda_work[A_eq.shape[0]:]

    stationarity = G @ x_opt + c - A_work.T @ lambda_work
    converged_flag = (l2_norm(stationarity) <= np.sqrt(conv_tol) and
                      l2_norm(A_eq @ x_opt - b_eq) <= np.sqrt(conv_tol) and
                      np.all(A_ineq @ x_opt >= b_ineq - np.sqrt(conv_tol)) and
                      np.all(multipliers[A_eq.shape[0]:] >= -np.sqrt(conv_tol)))
    
    iterations = [
        {'iter': 0, 'x': x0, 'func': f(x0), 'grad': G @ x0 + c},
        {'iter': 1, 'x': x_opt, 'func': f(x_opt), 'grad': stationarity}
    ]
    if converged_flag:
        logger.info(f'{method} converged')
    else:
        warnings.warn(f'{method} did not converge in {max_iter} iterations', RuntimeWarning)
    return MinimizeResult(
        x=x_opt,
        f_min=f(x_opt),
        iterations=iterations,
        solver=method,
        method='QP',
        converged=converged_flag,
        extras={'lambdas': multipliers}
    )


QP_METHODS = {
    'projected cg': projected_cg,
    'active set': active_set,
    'interior point': interior_point
}