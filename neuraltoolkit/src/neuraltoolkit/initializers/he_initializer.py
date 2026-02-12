import numpy as np
from math import sqrt
from .initializer import Initializer

class He_initializer(Initializer):
    def __call__(self, fan_in:int, shape:tuple, *args, **kwargs):
        std = sqrt(2 / fan_in)
        return np.random.normal(loc=0, scale=std, size=shape)