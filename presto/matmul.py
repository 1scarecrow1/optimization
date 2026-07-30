import numpy as np

class Array:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.n1 = np.ndim(self.value)

    def __transpose__(self):
        if self.n1 == 0:
            raise TypeError('scalars cannot be transposed')
        elif self.n1 == 1:
            return self.value[:, np.newaxis]

    def __matmul__(self, other):
        n2 = np.ndim(other.value)
        if self.n1 == 0:
            return self.value * other.value 
        elif self.n1 == 1:
            if n2 == 0:
                return n2 * self.n1 
            elif n2 == 1:
                return np.outer(self.value, other.value)





