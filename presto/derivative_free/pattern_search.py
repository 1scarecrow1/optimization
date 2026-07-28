from functools import partial
import numpy as np
from presto.trust_region.trust_region import reduction_ratio

def pattern_search(func, x, g_cur, B_cur, a0, D0, cont_factor, expand_factor, conv_tol, model, suff_decr_func=reduction_ratio, max_iter=200):
    alpha = a0
    x_cur = np.array(x, dtype=float)
    f_cur = func(x_cur)
    rho = partial(suff_decr_func, f_cur)
    m = partial(model, f_cur, g_cur, B_cur)
    p = D0[0] # find p in D with most reduction
    for _ in range(max_iter):
        step = alpha*p
        if alpha <= conv_tol:
            return x_cur, step 
        x_test = x_cur + step 
        f_next = func(x_test)
        m_next = m(step)
        rho_cur = rho(f_next, m_next)
        if f_next < f_cur - rho_cur:
            x_cur = x_cur + step 
            alpha *= expand_factor
        else:
            alpha *= cont_factor  
    print(f'Failed to converge in {max_iter} iterations')
    return x_cur, step 

def coordinate_direction_set(n, neg=True):
    # D = set() # or matrix 
    # for i in range(n):
    #     e = np.zeros(n)
    #     e[i] = 1
    #     D.add(e)
    #     D.add(-e)
    I = np.identity(n)
    if not neg:
        return I
    D = np.hstack((I, -I))
    return D 

def direction_set(n):
    '''
    p_i = (1/2n)e - e_i for i=1,2,..,n, p_n+1 = (1/2n)e
    '''
    D = np.ones((n, n+1)) / (2*n)
    D[np.diag_indices(n)] -= 1 # or D - I(n)
    return D 



