import pytest
from presto.linalg import *

QP_G = 2 * np.identity(2)
QP_C = np.array([-2.0, -5.0])
QP_A = np.array([[1.0, -2.0], [-1.0, -2.0], [-1.0, 2.0],
                 [1.0, 0.0], [0.0, 1.0]])
QP_B = np.array([-2.0, -6.0, -2.0, 0.0, 0.0])

def test_damped_bfgs_preserves_positive_definiteness():
    from presto.solvers.quasi_newton import damped_bfgs
    x_cur = np.zeros(2)
    x_next = np.array([1.0, 0.0])
    g_cur = np.zeros(2)
    g_next = np.array([-1.0, 0.0])
    B_next = damped_bfgs(x_cur, g_cur, x_next, g_next, np.identity(2))
    assert np.all(np.linalg.eigvalsh(B_next) > 0)

def test_projected_cg_solves_equality_constrained_qp():
    from presto.conjugate_gradient.conjugate_gradient import projected_cg
    G = np.array([[6.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 4.0]])
    c = np.array([-8.0, -3.0, -3.0])
    A = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    b = np.array([3.0, 0.0])
    x_opt = projected_cg(G, c, A, b, np.array([3.0, 0.0, 0.0]),
                         conv_tol=1e-10)
    assert np.allclose(x_opt, [2.0, -1.0, 1.0])

@pytest.mark.parametrize('method', ['active set', 'interior point'])
def test_quadratic_programming_methods(method):
    from presto.quadratic_programming.optimize import minimize_quadratic
    result = minimize_quadratic(QP_G, QP_C, x=np.array([2.0, 0.0]),
                                A_ineq=QP_A, b_ineq=QP_B, method=method)
    assert result.converged
    assert np.allclose(result.x, [1.4, 1.7], atol=1e-6)

@pytest.mark.parametrize('method', ['augmented lagrangian', 'line search',
                                    'trust region'])
def test_constrained_nonlinear_methods(method):
    from presto.quadratic_programming.optimize import minimize
    func = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2
    grad = lambda x: 2 * (x - np.array([1.0, 2.0]))
    constraints = lambda x: np.array([x.sum() - 1])
    jac = lambda x: np.array([[1.0, 1.0]])
    args = ({'equalities': constraints, 'equality_jac': jac}
            if method == 'line search' else {'constraints': constraints, 'jac': jac})
    result = minimize(func, np.array([2.0, 2.0]), method=method,
                      grad=grad, conv_tol=1e-6, max_iter=50, **args)
    assert result.converged
    assert np.allclose(result.x, [0.0, 1.0], atol=1e-5)

def main():
    L = np.array([[3, 0, 0, 0], [-1, 5, 0, 0], [2, 1, 4, 0], [7, 0, 3, 6]])
    A = L @ L.T
    print(A)
    print(np.linalg.eigvalsh(A))
    print(ssor(A))
    B, C = incomplete_cholesky(A)
    print(np.linalg.inv(A))
    print(B)
    print(C)

if __name__ == "__main__":
    main()
