import numpy as np
from presto.conjugate_gradient.conjugate_gradient import projected_cg
from presto.linalg import l2_norm
from presto.quadratic_programming.active_set import active_set, feasible_point
from presto.solvers.quasi_newton import damped_bfgs
from presto.trust_region.trust_region import trust_region_subproblem

def line_search_sqp(func, x, equalities, inequalities, grad,
                    equality_jac, inequality_jac, hess=None, B0=None,
                    lambdas=None, eta=0.1, tau=0.5, rho=0.5,
                    conv_tol=1e-6, max_iter=100):
    x_cur = x.copy()
    n = x_cur.size
    ceq = equalities(x_cur)
    cineq = inequalities(x_cur)
    m_eq = ceq.size

    lambda_cur = np.zeros(m_eq + cineq.size) if lambdas is None else lambdas.copy()
    B_cur = np.identity(n) if B0 is None else B0.copy()
    penalty = max(1.0, np.max(np.abs(lambda_cur)) + 1.0 if lambda_cur.size else 1.0)

    for _ in range(max_iter):
        f_cur = func(x_cur)
        g_cur = grad(x_cur)
        ceq = equalities(x_cur)
        cineq = inequalities(x_cur)
        A_eq = equality_jac(x_cur)
        A_ineq = inequality_jac(x_cur)
        stationarity = g_cur - A_eq.T @ lambda_cur[:m_eq] - A_ineq.T @ lambda_cur[m_eq:]

        optimality = max(l2_norm(stationarity), l2_norm(ceq),
                         l2_norm(np.maximum(-cineq, 0.0)),
                         l2_norm(lambda_cur[m_eq:] * cineq))
        if optimality <= conv_tol and np.all(lambda_cur[m_eq:] >= -conv_tol):
            return x_cur

        if hess is not None:
            B_cur = hess(x_cur, lambda_cur)

        p_start = feasible_point(A_eq, -ceq, A_ineq, -cineq,
                                 np.zeros(n), conv_tol=conv_tol)
        p_cur = active_set(B_cur, g_cur, A_eq, -ceq, A_ineq, -cineq,
                           p_start, conv_tol=min(conv_tol, 1e-8),
                           max_iter=max(50, 10*n))

        active = np.isclose(A_ineq @ p_cur, -cineq, atol=conv_tol)
        A_work = np.vstack((A_eq, A_ineq[active]))

        lambda_work = np.linalg.lstsq(A_work.T, B_cur @ p_cur + g_cur, rcond=None)[0]
        lambda_next = np.zeros_like(lambda_cur)
        lambda_next[:m_eq] = lambda_work[:m_eq]
        lambda_next[m_eq:][active] = lambda_work[m_eq:]

        violation_cur = constraint_violation(ceq, cineq)
        violation_model = constraint_violation(ceq + A_eq @ p_cur,
                                                cineq + A_ineq @ p_cur)
        violation_reduction = violation_cur - violation_model

        model_change = g_cur @ p_cur + 0.5 * p_cur @ B_cur @ p_cur

        if violation_reduction > np.finfo(float).eps:
            penalty = max(penalty, model_change / ((1 - rho) * violation_reduction) + 1e-8)
        if lambda_next.size:
            penalty = max(penalty, np.max(np.abs(lambda_next)) + 1e-8)

        merit_cur = f_cur + penalty * violation_cur
        directional = g_cur @ p_cur - penalty * violation_reduction
        alpha = 1.0

        for _ in range(50):
            x_next = x_cur + alpha * p_cur
            merit_next = (func(x_next) + penalty *
                          constraint_violation(equalities(x_next), inequalities(x_next)))
            if merit_next <= merit_cur + eta * alpha * directional:
                break
            alpha = tau * alpha

        g_next = grad(x_next)
        A_eq_next = equality_jac(x_next)
        A_ineq_next = inequality_jac(x_next)

        if hess is None:
            lag_grad_cur = g_cur - A_eq.T @ lambda_next[:m_eq] - A_ineq.T @ lambda_next[m_eq:]
            lag_grad_next = g_next - A_eq_next.T @ lambda_next[:m_eq] - A_ineq_next.T @ lambda_next[m_eq:]
            B_cur = damped_bfgs(x_cur, lag_grad_cur, x_next,
                                lag_grad_next, B_cur)

        x_cur = x_next
        lambda_cur = lambda_next

    return x_cur

def trust_region_sqp(func, x, constraints, grad, jac, hess=None,
                     B0=None, lambdas=None, rad0=1.0, rad_max=np.inf,
                     zeta=0.8, eta=0.1, gamma=0.25, rho=0.5,
                     conv_tol=1e-6, max_iter=100):
    x_cur = x.copy()
    n = x_cur.size
    c_cur = constraints(x_cur)
    lambda_cur = np.zeros(c_cur.size) if lambdas is None else lambdas.copy()
    B_cur = np.identity(n) if B0 is None else B0.copy()
    rad_cur = rad0
    penalty = 1.0

    for _ in range(max_iter):
        f_cur = func(x_cur)
        g_cur = grad(x_cur)
        c_cur = constraints(x_cur)
        A_cur = jac(x_cur)
        lambda_cur = np.linalg.lstsq(A_cur.T, g_cur, rcond=None)[0]
        stationarity = g_cur - A_cur.T @ lambda_cur

        if max(l2_norm(stationarity), l2_norm(c_cur)) <= conv_tol:
            return x_cur

        normal_grad = A_cur.T @ c_cur
        if l2_norm(normal_grad) <= conv_tol:
            v_cur = np.zeros(n)
        else:
            v_cur = trust_region_subproblem(normal_grad, A_cur.T @ A_cur,
                                             zeta * rad_cur)

        if hess is not None:
            B_cur = hess(x_cur, lambda_cur)

        p_cur = projected_cg(B_cur, g_cur, A_cur, A_cur @ v_cur, v_cur,
                             radius=rad_cur, conv_tol=min(conv_tol, 1e-8),
                             max_iter=n)

        violation_cur = np.abs(c_cur).sum()
        violation_model = np.abs(c_cur + A_cur @ p_cur).sum()
        violation_reduction = violation_cur - violation_model
        model_change = g_cur @ p_cur + 0.5 * p_cur @ B_cur @ p_cur

        if violation_reduction > np.finfo(float).eps:
            penalty = max(penalty, model_change / ((1 - rho) * violation_reduction) + 1e-8)

        predicted = -model_change + penalty * violation_reduction
        x_trial = x_cur + p_cur
        c_trial = constraints(x_trial)

        actual = f_cur + penalty * violation_cur - func(x_trial) - penalty * np.abs(c_trial).sum()
        ratio = actual / predicted if predicted > np.finfo(float).eps else -np.inf

        if ratio > eta:
            g_next = grad(x_trial)
            A_next = jac(x_trial)
            lambda_next = np.linalg.lstsq(A_next.T, g_next, rcond=None)[0]

            if hess is None:
                lag_grad_cur = g_cur - A_cur.T @ lambda_next
                lag_grad_next = g_next - A_next.T @ lambda_next
                B_cur = damped_bfgs(x_cur, lag_grad_cur, x_trial,
                                    lag_grad_next, B_cur)

            x_cur = x_trial
            lambda_cur = lambda_next

            if ratio > 0.75 and l2_norm(p_cur) >= 0.8 * rad_cur:
                rad_cur = min(2 * rad_cur, rad_max)
        else:
            rad_cur = gamma * (l2_norm(p_cur) if l2_norm(p_cur) > 0 else rad_cur)

    return x_cur

def constraint_violation(equalities, inequalities):
    return np.abs(equalities).sum() + np.maximum(-inequalities, 0.0).sum()

SQP_METHODS = {
    'line search': line_search_sqp,
    'trust region': trust_region_sqp,
    'byrd omojokun': trust_region_sqp
}
