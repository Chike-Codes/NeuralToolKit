import numpy as np
from ..optimizers.optimizer import Optimizer
from .layer import Layer

class Flatten(Layer):
    def __init__(self, input_shape:tuple=None):
        self.name = "flatten"
        self.input_shape = input_shape
        self.weights = np.array([])
        self.biases = np.array([])

    def get_inputDim(self):
        return (self.input_shape,)
    
    def get_outputDim(self):
        return (self.output_shape,)
    
    def initialize_parameters(self):
        self.output_shape = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
    
    def initialize_optimizers(self, optimizer:Optimizer):
        self.weight_optimizer = None
        self.bias_optimizer = None
    
    def optimize_parameters(self, weight_gradient, bias_gradient):
        pass


    def forward(self, x):
        return x.reshape((x.shape[0], self.output_shape)), np.ones((x.shape[0], self.output_shape))
    
    def back(self, error, *args, **kwargs):
        d_input = np.reshape(error, (error.shape[0], self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        return d_input
    
    def calc_gradients(self, *args, **kwargs):
        return np.array([]), np.array([])
    
    def get_config(self):
        config = {
            "input_shape": self.input_shape
        }
        return config