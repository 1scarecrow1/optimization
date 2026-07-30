from collections.abc import Callable
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MinimizeResult:
    x: np.ndarray
    f_min: float | np.ndarray
    iterations: list[dict]
    func: Callable | None = None
    solver: Callable | str | None = None
    method: Callable | str | None = None       
    converged: bool = False
    extras: dict = field(default_factory=dict)

    @property
    def terminal(self):
        return self.iterations[-1]['iter'] if self.iterations else 0


