from .activation import Activation
import numpy as np
class Relu(Activation):
    def __call__(self, x):
        res = np.maximum(0, x)
        return res    
    
    def derive(self, x):
        return (x > 0).astype(int)