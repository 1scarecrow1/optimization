import numpy as np
from functools import partial
from presto.linalg import l2_norm

def active_set(G, c, A_eq, b_eq, A_ineq, b_ineq, x,
               working_set=None, conv_tol=1e-8, max_iter=100):
    x_cur = x.copy()
    if working_set is None:
        working = list(np.flatnonzero(
            np.isclose(A_ineq @ x_cur, b_ineq, atol=conv_tol)))
    else:
        working = list(working_set)

    for _ in range(max_iter):
        g_cur = G @ x_cur + c
        A_work = np.vstack((A_eq, A_ineq[working]))

        if A_work.shape[0]:
            K = np.block([[G, -A_work.T],
                          [A_work, np.zeros((A_work.shape[0], A_work.shape[0]))]])
            rhs = np.concatenate((-g_cur, np.zeros(A_work.shape[0])))
            solution = np.linalg.lstsq(K, rhs, rcond=None)[0]
            p_cur = solution[:x_cur.size]
            lambdas = solution[x_cur.size:]
        else:
            p_cur = np.linalg.lstsq(G, -g_cur, rcond=None)[0]
            lambdas = np.empty(0)

        if l2_norm(p_cur) <= conv_tol:
            lambda_ineq = lambdas[A_eq.shape[0]:]
            if not lambda_ineq.size or np.all(lambda_ineq >= -conv_tol):
                return x_cur
            del working[int(np.argmin(lambda_ineq))]
            continue

        alpha = 1.0
        blocking = None

        for i in range(A_ineq.shape[0]):
            if i in working:
                continue
            aip = A_ineq[i] @ p_cur
            if aip < -conv_tol:
                step = (b_ineq[i] - A_ineq[i] @ x_cur) / aip
                if step < alpha:
                    alpha = max(0.0, step)
                    blocking = i

        x_cur = x_cur + alpha * p_cur
        if blocking is not None:
            A_next = np.vstack((A_work, A_ineq[blocking]))
            if np.linalg.matrix_rank(A_next, tol=conv_tol) > np.linalg.matrix_rank(A_work, tol=conv_tol):
                working.append(blocking)

    return x_cur

def feasible_point(A_eq, b_eq, A_ineq, b_ineq, x, conv_tol=1e-8,
                   max_iter=1000):
    x_cur = x.copy()
    if A_eq.shape[0]:
        x_cur = x_cur + np.linalg.lstsq(A_eq, b_eq - A_eq @ x_cur,
                                        rcond=None)[0]

    for _ in range(max_iter):
        feasible = True
        for i in range(A_ineq.shape[0]):
            violation = b_ineq[i] - A_ineq[i] @ x_cur
            if violation <= conv_tol:
                continue

            feasible = False
            direction = A_ineq[i].copy()

            if A_eq.shape[0]:
                multipliers = np.linalg.lstsq(A_eq @ A_eq.T,
                                              A_eq @ direction, rcond=None)[0]
                direction = direction - A_eq.T @ multipliers

            curvature = direction @ direction
            if curvature > np.finfo(float).eps:
                x_cur = x_cur + violation * direction / curvature
                
        if feasible:
            return x_cur

    return x_cur

def frank_wolfe(func, x0, grad, A, max_iter=150, **func_args):
    x = np.array(x0, dtype=float)
    g = partial(grad, A, **func_args)
    gx = g(x)
    k = 0

    while k < max_iter and not is_frank_wolfe_duality_gap_min(gx, x): 
        s_min = np.argmin(gx) 
        v = A @ x 
        a = (x @ v - v[s_min]) / (x @ v - 2*v[s_min] + A[s_min, s_min])
        x *= 1 - a 
        x[s_min] += a
        gx = g(x)
        k += 1

    return x

def is_frank_wolfe_duality_gap_min(g_cur, x_cur, eps=1e-6):
    return g_cur @ x_cur - np.min(g_cur) <= eps
