import numpy as np
from .optimizer import Optimizer

class Adam:
    def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999):
        # moving average decay for weights and biases for both first and second moments
        self.momentum = []
        self.velocity = []
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8
        self.time = 0

    def build(self, gradient_shape:tuple):
        self.momentum = np.zeros(gradient_shape)
        self.velocity = np.zeros(gradient_shape)

    def init(self):
        self.time = 0
        self.momentum *= 0
        self.velocity *= 0

    # updates weights and biases using Adagrad an adaptive optimizer that updates the LR for each parameter based on the sqrt of summed squares of previous gradients
    def optimize(self, gradient):
        self.time += 1
        self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * gradient

        self.velocity = self.beta2 * self.velocity + (1 - self.beta2) * (gradient ** 2)

        #bias corrected estimates
        EM = self.momentum / (1 - self.beta1 ** self.time)
        EV = self.velocity / (1 - self.beta2 ** self.time)

        return self.learning_rate * (EM / (np.sqrt(EV) + self.epsilon))