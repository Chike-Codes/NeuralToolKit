import numpy as np
from ..optimizers.optimizer import Optimizer
from .layer import Layer

class Concatenate(Layer):
    def __init__(self, input_shape:int=None, units:int=None):
        self.name = "concatenate"
        self.input_shape = input_shape
        self.concat_shape = units
        self.weights = np.array([])
        self.biases = np.array([])

    def get_inputDim(self):
        return (self.input_shape,)
    
    def get_outputDim(self):
        return (self.output_shape,)
    
    def get_concatDim(self):
        return (self.concat_shape,)
    
    def initialize_parameters(self):
        self.input_shape += self.concat_shape
        self.output_shape = self.input_shape
    
    def initialize_optimizers(self, optimizer:Optimizer):
        self.weight_optimizer = None
        self.bias_optimizer = None
    
    def optimize_parameters(self, weight_gradient, bias_gradient):
        pass


    def forward(self, x):
        derivatives = np.ones(self.output_shape)
        return x, derivatives
    
    def back(self, error, *args, **kwargs):
        d_input = error[:, :-self.concat_shape]
        return d_input
    
    def calc_gradients(self, *args, **kwargs):
        return np.array([]), np.array([])
    
    def get_config(self):
        config = {
            "input_shape": self.input_shape
        }
        return config