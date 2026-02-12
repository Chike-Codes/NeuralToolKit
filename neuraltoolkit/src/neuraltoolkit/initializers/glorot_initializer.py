import numpy as np
from math import sqrt
from .initializer import Initializer

class Glorot_initializer(Initializer):
    def __call__(self, fan_in:int, fan_out:int, shape:tuple):
        std = sqrt(2 / (fan_in + fan_out))
        return np.random.normal(loc=0, scale=std, size=shape)