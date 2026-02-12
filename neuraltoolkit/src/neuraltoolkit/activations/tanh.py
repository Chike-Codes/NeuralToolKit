from .activation import Activation
import numpy as np
class Tanh(Activation):
    def __call__(self, x):
        return np.tanh(x)
    
    def derive(self, x):
        return 1 - np.power(np.tanh(x), 2)