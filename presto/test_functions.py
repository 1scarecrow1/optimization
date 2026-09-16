import numpy as np

def rosenbrock(x, a=1, b=100):
    return np.sum(b*(x[1:] - x[:-1]**2)**2 + (a - x[:-1])**2)

def rosenbrock_gradient(x, a=1, b=100):
    g = np.empty_like(x)
    g[0] = -4*b*x[0]*(x[1] - x[0]**2) - 2*(a - x[0])
    g[-1] = 2*b*(x[-1] - x[-2]**2)
    g[1:-1] = 2*b * (x[1:-1] - x[:-2]**2) - 4*b*x[1:-1]*(x[2:] - x[1:-1]**2) - 2*(a - x[1:-1])
    return g

def rosenbrock_jacobian(x, a=1, b=100):
    return rosenbrock_gradient(x, a=a, b=b)

def rosenbrock_hessian(x, a=1, b=100):
    n = len(x)
    H = np.zeros((n, n), dtype=float)
    idx = np.arange(n)
    H[0, 0] = 12*b*x[0]**2 - 4*b*x[1] + 2
    H[1:-1, 1:-1][np.diag_indices(n-2)] = (
        12*b*x[1:-1]**2 - 4*b*x[2:] + 2 + 2*b
    )
    H[-1, -1] = 2*b

    H[idx[:-1], idx[1:]] = -4*b*x[:-1]
    H[idx[1:], idx[:-1]] = -4*b*x[:-1]

    return H


rosenbrock.gradient = rosenbrock_gradient
rosenbrock.jacobian = rosenbrock_jacobian
rosenbrock.hessian = rosenbrock_hessian
