import numpy as np
from .optimizer import Optimizer

class Adagrad:
    def __init__(self, learning_rate = 0.001):
        self.learning_rate = learning_rate
        self.epsilon = 1e-8
        

    def build(self, gradient_shape:tuple):
            self.learning_gradient = np.zeros(gradient_shape)

    def init(self):
        self.learning_gradient *= 0

    # updates weights and biases using Adagrad an adaptive optimizer that updates the LR for each parameter based on the sqrt of summed squares of previous gradients
    def optimize(self, gradient):
        self.learning_gradient += gradient ** 2
        return (self.learning_rate / (np.sqrt(self.learning_gradient) + self.epsilon)) * gradient