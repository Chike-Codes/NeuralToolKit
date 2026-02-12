from .activation import Activation
import numpy as np
class Softmax(Activation):
    def __call__(self, x):
        expos = np.exp(x - np.max(x, axis=1, keepdims=True))
        expoSum = np.sum(expos, axis=1, keepdims=True)
        res = np.zeros(x.shape)
        return expos / expoSum 
    
    def derive(self, x):
        return np.power(x, 0)