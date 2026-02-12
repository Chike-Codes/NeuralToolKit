import numpy as np
from ..initializers.initializer import Initializer
from ..activations.activation import Activation
from ..optimizers.optimizer import Optimizer
from .layer import Layer
from copy import deepcopy

class Dense(Layer):
    
    def __init__(self, activation:Activation, input_shape:int=None, output_shape:int=None, initializer=Initializer):
        self.name = "dense"
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.initializer = initializer

        self.activation = activation

    def get_inputDim(self):
        return (self.input_shape,)
    
    def get_outputDim(self):
        return (self.output_shape,)

    def initialize_parameters(self):
        self.weights = self.initializer(fan_in=self.input_shape, fan_out=self.output_shape, shape=(self.output_shape, self.input_shape))
        self.biases = np.zeros((1, self.output_shape))
    
    def initialize_optimizers(self, optimizer:Optimizer):
        self.weight_optimizer = optimizer
        self.bias_optimizer = deepcopy(optimizer)

        self.weight_optimizer.build(self.weights.shape)
        self.bias_optimizer.build(self.biases.shape)
    
    def optimize_parameters(self, weight_gradient, bias_gradient):
        self.weights -= self.weight_optimizer.optimize(weight_gradient)
        self.biases -= self.bias_optimizer.optimize(bias_gradient)
    
    def forward(self, x):
        z = x @ self.weights.T + self.biases
        return self.activation(z), self.activation.derive(z)
    
    def back(self, error, derivatives):
        #calculates the output error and graidents for the previous layer
        d_input = (error @ self.weights) * derivatives
        return d_input
        
    def calc_gradients(self, error, input):
        weight_gradient = error.T @ input
        bias_gradient = np.sum(error, axis=0)
        return weight_gradient, bias_gradient
    
    def logits(self, x):
        return x @ self.weights.T + self.biases
    
    def get_config(self):
        config = {
            "activation": self.activation,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape
        }
        return config
    
    def copy(self):
        layer = Dense(self.input_shape, self.output_shape, self.activation)
        layer.weights = self.weights.copy()
        layer.biases = self.biases.copy()
        return layer