import numpy as np
from .optimizer import Optimizer

class RMSProp:
    def __init__(self, learning_rate = 0.001, decay_rate = 0.9):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = 1e-8
        

    def build(self, gradient_shape:tuple):
        self.moving_average = np.zeros(gradient_shape)

    def init(self):
        self.moving_average *= 0

    # updates weights and biases using Adagrad an adaptive optimizer that updates the LR for each parameter based on the sqrt of summed squares of previous gradients
    def optimize(self, gradient):
        self.moving_average = self.decay_rate * self.moving_average + (1 - self.decay_rate) * gradient ** 2
        return(self.learning_rate / (np.sqrt(self.moving_average) + self.epsilon)) * gradient