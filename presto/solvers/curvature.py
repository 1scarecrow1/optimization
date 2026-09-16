import numpy as np 

class Curvature:
    """
    Curvature of the local model.

        dense  : B is an (n,n) symmetric matrix
        matvec : B is a callable p -> Bp          (Hessian-free: Newton-CG/Lanczos, L-BFGS)
        jac    : B is an (m,n) Jacobian J with residual r
                 -> g = Jᵀr,  B = JᵀJ,  pᵀBp = ‖Jp‖²    (Gauss-Newton / LM)

    Solvers ask for the primitive they need; `.dense` raises a useful error
    when only products are available.
    """
    __slots__ = ('B', 'kind', 'resid', 'n')

    def __init__(self, B, resid=None, kind=None):
        self.resid = resid
        if kind is not None:
            self.kind = kind
        elif callable(B) and not isinstance(B, np.ndarray):
            self.kind = 'matvec'
        elif resid is not None:
            self.kind = 'jac'
        else:
            self.kind = 'dense'
        self.B = B if self.kind == 'matvec' else np.asarray(B, dtype=float)
        self.n = None if self.kind == 'matvec' else self.B.shape[1]

    def matvec(self, p):
        if self.kind == 'dense':  return self.B @ p
        if self.kind == 'matvec': return self.B(p)
        return self.B.T @ (self.B @ p)            # JᵀJp — never forms JᵀJ

    def quad(self, p):                            # pᵀBp
        if self.kind == 'jac':
            Jp = self.B @ p
            return Jp @ Jp                        # ‖Jp‖², O(mn) not O(mn²)
        return p @ self.matvec(p)

    @property
    def dense(self):
        if self.kind == 'dense': return self.B
        if self.kind == 'jac':   return self.B.T @ self.B
        raise TypeError(
            "this trust-region subproblem solver needs an explicit matrix, but the "
            "curvature was supplied as a matrix-vector product. Use a product-only "
            "method (cauchy_point, cg_steihaug, newton_lanczos), or pass hess=... "
            "instead of hessp=...")

    def scaled(self, D):
        """B -> D⁻¹BD⁻¹ for the elliptical region ‖Dp‖ ≤ Δ."""
        if D is None:
            return self
        d_inv = 1.0 / (np.diag(D) if np.ndim(D) == 2 else np.asarray(D, float))
        if self.kind == 'matvec':
            return Curvature(lambda p: d_inv * self.B(d_inv * p), kind='matvec')
        if self.kind == 'jac':
            return Curvature(self.B * d_inv, resid=self.resid, kind='jac')   # J D⁻¹
        return Curvature(self.B * np.outer(d_inv, d_inv), kind='dense')
