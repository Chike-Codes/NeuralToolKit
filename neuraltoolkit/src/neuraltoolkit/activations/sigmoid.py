from .activation import Activation
import numpy as np
class Sigmoid(Activation):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derive(self, x):
        return self(x) * (1 - self(x))