import numpy as np
from presto.linalg import l2_norm

def interior_point(G, c, A, b, x, y, lambdas,
                   tau=0.995, conv_tol=1e-8, max_iter=100):
    x_cur = x.copy()
    y_cur = y.copy()
    lambda_cur = lambdas.copy()
    m, n = A.shape

    for _ in range(max_iter):
        r_dual = G @ x_cur + c - A.T @ lambda_cur
        r_primal = A @ x_cur - y_cur - b
        
        mu = y_cur @ lambda_cur / m
        if max(l2_norm(r_dual), l2_norm(r_primal), mu) <= conv_tol:
            return x_cur

        Y = np.diag(y_cur)
        Lambda = np.diag(lambda_cur)

        K = np.block([[G, np.zeros((n, m)), -A.T],
                      [A, -np.identity(m), np.zeros((m, m))],
                      [np.zeros((m, n)), Lambda, Y]])
        rhs = -np.concatenate((r_dual, r_primal, y_cur * lambda_cur))
        affine = np.linalg.lstsq(K, rhs, rcond=None)[0]

        dy_aff = affine[n:n + m]
        dlambda_aff = affine[n + m:]

        alpha_aff_primal = min(1.0, np.min(-y_cur[dy_aff < 0] / dy_aff[dy_aff < 0])
                               if np.any(dy_aff < 0) else 1.0)

        alpha_aff_dual = min(1.0, np.min(-lambda_cur[dlambda_aff < 0] / dlambda_aff[dlambda_aff < 0])
                             if np.any(dlambda_aff < 0) else 1.0)

        mu_aff = ((y_cur + alpha_aff_primal * dy_aff) @
                  (lambda_cur + alpha_aff_dual * dlambda_aff) / m)

        sigma = (mu_aff / mu)**3
        correction = y_cur * lambda_cur + dy_aff * dlambda_aff - sigma * mu
        rhs = -np.concatenate((r_dual, r_primal, correction))

        direction = np.linalg.lstsq(K, rhs, rcond=None)[0]
        dx = direction[:n]
        dy = direction[n:n + m]
        dlambda = direction[n + m:]

        alpha_primal = min(1.0, tau * np.min(-y_cur[dy < 0] / dy[dy < 0])
                           if np.any(dy < 0) else 1.0)
        alpha_dual = min(1.0, tau * np.min(-lambda_cur[dlambda < 0] / dlambda[dlambda < 0])
                         if np.any(dlambda < 0) else 1.0)

        x_cur = x_cur + alpha_primal * dx
        y_cur = y_cur + alpha_primal * dy
        lambda_cur = lambda_cur + alpha_dual * dlambda

    return x_cur
