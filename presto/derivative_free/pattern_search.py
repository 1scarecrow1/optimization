from functools import partial
import numpy as np
from presto.trust_region.trust_region import reduction_ratio

def pattern_search(
    func, x, g_cur, B_cur, a0, D0, contract_factor, expand_factor,
    conv_tol, model, suff_decr_func=reduction_ratio, max_iter=200
):
    alpha = a0
    x_cur = np.array(x, dtype=float)
    f_cur = func(x_cur)

    rho = partial(suff_decr_func, f_cur)
    m = partial(model, f_cur, g_cur, B_cur)

    p = D0[0]  # find direction with most reduction

    for _ in range(max_iter):
        step = alpha * p

        if alpha <= conv_tol:
            return x_cur, step

        x_test = x_cur + step
        f_next = func(x_test)
        m_next = m(step)
        rho_cur = rho(f_next, m_next)

        if f_next < f_cur - rho_cur:
            x_cur = x_test
            f_cur = f_next
            alpha *= expand_factor
        else:
            alpha *= contract_factor

    print(f"Failed to converge in {max_iter} iterations")
    return x_cur, step

def coordinate_direction_set(
    n: int,
    neg: bool = True,
    coord_set: set | np.ndarray | None = None,
):
    """
    Create or extend a set/matrix of coordinate direction vectors

    Parameters
    ----------
    n : int
        Dimension of the coordinate space.
    neg : bool, default=True
        Whether to include negative coordinate directions.
    coord_set : set | np.ndarray | None, default=None
        Existing direction set/matrix. If None, a new matrix is created.

    Returns
    -------
    set | np.ndarray
        Coordinate direction vectors.
    """

    if isinstance(coord_set, set):
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1
            coord_set.add(tuple(e))
            if neg:
                coord_set.add(tuple(-e))
        return coord_set

    if isinstance(coord_set, np.ndarray):
        I = np.identity(n)
        directions = I if not neg else np.hstack((I, -I))
        return np.hstack((coord_set, directions))

    I = np.identity(n)
    if not neg:
        return I
        
    return np.hstack((I, -I))

def positive_spanning_set(n):
    '''
    p_i = (1/2n)e - e_i for i=1,2,..,n, p_n+1 = (1/2n)e
    '''
    D = np.ones((n, n+1)) / (2*n)
    D[np.diag_indices(n)] -= 1 # or D - I(n)
    return D 



