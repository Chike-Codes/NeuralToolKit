import numpy as np
from .optimizer import Optimizer

class Sgd:
    def __init__(self, learning_rate = 0.001, momentum_rate = 0.9):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate        

    def build(self, gradient_shape:tuple):
        self.velocity = np.zeros(gradient_shape)

    def init(self):
        self.velocity *= 0

    # updates weights and biases using stochastic gradient descent with momentum
    def optimize(self, gradient):
        self.velocity = (self.momentum_rate * self.velocity + (1 - self.momentum_rate) * gradient) * self.learning_rate
        return self.velocity