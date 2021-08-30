import numpy as np

class RandomLinearModel:
    def __init__(self, ndims, mask=None, fn=np.sign):
        self.w = np.random.uniform(size=(ndims,))
        self.mask = mask
        self.fn = fn

    def predict(self, x):
        if self.mask is None:
            return self.fn(np.matmul(x, self.w))
        else:
            return self.fn(np.matmul(x * self.mask, self.w))