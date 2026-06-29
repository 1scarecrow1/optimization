import numpy as np

def l2_norm(x):
    if x.ndim == 0:
        return np.abs(x)
    elif x.ndim == 1:
        return np.sqrt(x @ x)
    elif x.ndim == 2:
        eigs = np.linalg.eig(x.T@x)
        return np.sqrt(np.max(eigs))
    else:
        raise NotImplementedError(f"not defined for {x.ndim}-dimensional object yet")