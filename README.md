# presto

## Numerical optimization in Python, following Nocedal & Wright

Line search, trust region, conjugate gradient, least squares and quadratic programming methods

Allows for more configuration than SciPy. The descent direction, step-length, Hessian modification, 
trust-region subproblem solver and the radius update thresholds are all separate arguments, and 
each one takes either an existing implementation in presto or a function of your own, where compatible.

## Install

```
pip install presto-opt
```

The distribution is `presto-opt`; the import name is `presto`.

From source:

```
git clone https://github.com/1scarecrow1/optimization.git
cd optimization
pip install -e .
```

Python 3.10+, numpy, scipy and matplotlib.

## Quick start

```python
import numpy as np
from presto.line_search.optimize import minimize
from presto.test_functions import rosenbrock

res = minimize(rosenbrock, np.array([-1.2, 1.0]), 'bfgs', 'wolfe',
               solver_args={'inv': True},
               line_search_args={'c1': 1e-4, 'c2': 0.9, 'interp_method': 'cubic'},
               conv_tol=1e-6, max_iter=200)

res.x          # array([1., 1.])
res.f_min      # 2.89e-16
res.converged  # True
res.iterations # per-iteration x, f, grad, step
```

The direction, line search and interpolation methods can be independently chosen:

```python
minimize(rosenbrock, np.array([1.2, 1.2]), 'sr1', 'backtracking',
         solver_args={'inv': True},
         line_search_args={'a0': 1.0, 'c1': 1e-4, 'rho': 0.5},
         conv_tol=1e-6, max_iter=500)
```

Trust region, with modifyable radius update thresholds:

```python
import presto.solvers.newton as nu
from presto.trust_region.optimize import minimize
from presto.trust_region.trust_region import dogleg

res = minimize(rosenbrock, np.array([-1.2, 1.0]), nu.newton, dogleg,
               solver_args={'eta': 0.15, 'rad_max': 5.0,
                            'thresholds': {'rho_min': 0.2, 'rho_max': 0.8,
                                           'contraction': 0.25, 'expansion': 2.0}},
               trust_region_args={'rad0': 1.0}, conv_tol=1e-6)
```

Nonlinear CG, choosing the β formula:

```python
from presto.conjugate_gradient.optimize import minimize

res = minimize(rosenbrock, np.array([-1.2, 1.0]), 'wolfe', 'HZ',
               restart=True, cg_restart_tol=1e-4, conv_tol=1e-6)
```

## Optimization Methods

Anything here can also be a callable with the same signature.

### Minimizers

| Function | Solves |
|---|---|
| `line_search.optimize.minimize` | unconstrained: descent direction plus step length |
| `trust_region.optimize.minimize` | unconstrained: model minimised inside a radius |
| `conjugate_gradient.optimize.minimize` | unconstrained: nonlinear CG |
| `inexact_newton.newton_cg.minimize` | unconstrained: Newton-CG with a line search |
| `least_squares.nonlinear.gauss_newton` / `.levenberg_marquardt` | nonlinear least squares |
| `quadratic_programming.optimize.minimize_quadratic` | QP with linear constraints |
| `quadratic_programming.optimize.minimize` | constrained nonlinear: SQP and augmented Lagrangian |

### Options

| Method | Options |
|---|---|
| Descent direction | `newton`, `modified_newton`, `bfgs`, `sr1`, `broyden`, `gradient descent` |
| Line search | `backtracking` / `armijo`, `wolfe` / `strong wolfe` |
| Zoom interpolation | `cubic`, `quadratic`, `bisection` |
| Initial step length | `constant_fo_change`, `quadratic_interpolation`, `cubic_interpolation`, `bisection` |
| Hessian modification | `regularise`, `regularised cholesky`, `inexact modified cholesky`, `incomplete cholesky`, `incomplete cholesky shifted`, `incomplete LU`, `gauss newton` |
| Quasi-Newton update | `bfgs`, `damped bfgs`, `sr1`, `broyden` |
| Trust-region subproblem | `dogleg`, `cholesky`, `qr`, `cauchy_point`, `generalised_cauchy`, `cg steihaug` |
| Linear CG | `cg`, `preconditioned cg`, `projected cg` |
| Nonlinear CG β | `FR`, `PR`, `HS`, `DY`, `HZ` |
| CG preconditioner | `incomplete_cholesky`, `incomplete_cholesky_shifted`, `incomplete_LU`, `ssor_preconditioner` |
| Linear least squares | `cholesky`, `qr`, `svd`, `cg` |
| Nonlinear least squares | `gauss_newton`, `levenberg_marquardt` |
| Loss functions | `squared`, `absolute`, `huber` |
| QP | `active set`, `interior point`, `projected cg`, `augmented and penalised Lagrangian` |
| SQP | `line search`, `trust region` |

`presto.linalg` has the supporting numerics: LU with and without pivoting, Cholesky, modified and
incomplete Cholesky, Givens QR, triangular solves, SSOR, and matrix property predicates.

## Compared to `scipy.optimize`

What you can change here that SciPy won't let you change, plus a few things it has no public
interface for at all.

**Line search parameters.** `scipy.optimize.minimize` has nowhere to put `c1`, `c2`, `rho`,
`a_max`, the backtracking cap or the zoom iteration cap. They exist on the standalone
`scipy.optimize.line_search`, but you can't put that into a `minimize` method. Here they can be 
specified in `line_search_args`, with validity checks for arg compatibility.

**Direction and step length are separate.** SciPy ties them to the method name. For instance
`BFGS` is tied to Wolfe. Any of the directions in presto can be specified with different line search
or trust region methods. 

**SR1 and Broyden as descent directions.** SciPy has SR1 only as a `HessianUpdateStrategy` for
`trust-constr`, and Broyden only as a root finder. Neither is available for line-search minimisation.

**7 ways to fix an indefinite Hessian**, from eigenvalue regularisation to shifted incomplete
Cholesky. This choice is not available at all in SciPy.

**Trust-region update thresholds.** SciPy's `dogleg`, `trust-ncg` and `trust-exact` take `eta`,
`initial_trust_radius` and `max_trust_radius`, but the accept/shrink/expand ratios
and the shrink and growth factors are fixed in the solver. Here they can be specified in `thresholds`,
and `expand_condition` is a callable you can pass.

**Scaled trust regions.** Unlike SciPy's trust-region methods which are spherical only, here you can 
specify a scaling matrix `D` to have elliptical trust regions.

**Five β formulas for nonlinear CG** SciPy's `method='CG'` is tied to Polak–Ribière+ only. 
In presto there are multiple choices for β updates, along with the choice to specify restarts.

**Preconditioner constructors.** In presto you can choose how to construct the preconditioner from: 
incomplete Cholesky, shifted incomplete Cholesky and SSOR.

**Projected CG** — CG in the null space of the equality constraints, for which there is no public counterpart in SciPy.

**Gauss–Newton.** `scipy.optimize.least_squares` gives you `trf`, `dogbox` and `lm`, but not Gauss–Newton.

**Quadratic programming.** SciPy has no public QP solver. In presto there are active-set methods, augmented 
and penalised Lagrangian methods, primal–dual interior-point methods, and SQP with line-search or trust-region methods. 

## Tests

```
pytest
```

## License

MIT.
