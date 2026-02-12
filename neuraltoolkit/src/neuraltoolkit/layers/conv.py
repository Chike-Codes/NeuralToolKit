import numpy as np
from ..initializers.initializer import Initializer
from ..activations.activation import Activation
from ..ops import convolve, im2col, up_sample
from ..optimizers.optimizer import Optimizer
from .layer import Layer
from copy import deepcopy

class Conv(Layer):
    def __init__(
            self, 
            activation:Activation,
            filters:int, 
            kernel_size:int, 
            input_shape:tuple=None, 
            stride:int=1, 
            padding:int=1, 
            initializer=Initializer
            ):
        self.name = "conv"
        self.input_shape = input_shape
        self.filter_count = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.output_shape = None

        self.initializer = initializer

        self.activation = activation

    def get_inputDim(self):
        return self.input_shape
    
    def get_outputDim(self):
        return self.output_shape

    def initialize_parameters(self):
        outputHight = (self.input_shape[1] - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (self.input_shape[2] - self.kernel_size + 2 * self.padding) // self.stride + 1
        self.output_shape = (self.filter_count, outputHight, output_width)

        self.weights = self.initializer(
            fan_in=self.input_shape[0] * self.kernel_size**2, 
            fan_out=self.filter_count * self.kernel_size**2,
            shape=(self.filter_count, self.input_shape[0], self.kernel_size, self.kernel_size))
        self.biases = np.zeros(self.filter_count)
    
    def initialize_optimizers(self, optimizer:Optimizer):
        self.weight_optimizer = optimizer
        self.bias_optimizer = deepcopy(optimizer)

        self.weight_optimizer.build(self.weights.shape)
        self.bias_optimizer.build(self.biases.shape)
    
    def optimize_parameters(self, weight_gradient, bias_gradient):
        self.weights -= self.weight_optimizer.optimize(weight_gradient.reshape(self.weights.shape))
        self.biases -= self.bias_optimizer.optimize(bias_gradient)

    def forward(self, x):
        z = convolve(x, self.weights, stride=self.stride, padding=self.padding, bias=self.biases)
        return self.activation(z), self.activation.derive(z)
    
    def back(self, error, derivatives, *args, **kwargs):
        #calculate dL/dx
        fliped_w = self.weights[:, :, ::-1, ::-1]
        transposed_w = fliped_w.transpose(1 ,0 ,2, 3)

        padding = self.kernel_size - 1 - self.padding
        up_sampled_x = up_sample(error, self.stride)
        
        d_input = convolve(up_sampled_x, transposed_w, stride=1, padding=padding)

        return d_input * derivatives
    
    def calc_gradients(self, error, input):
        #calculate dL/dw
        x_cols = im2col(input, self.kernel_size, self.stride, padding=self.padding)[0]
        dy_flat = error.reshape(error.shape[0], error.shape[1], -1)
        dw = dy_flat @ x_cols
        dw = np.sum(dw, axis=0)

        #calculate dL/db
        db = np.sum(dy_flat, axis=(0, 2))

        return dw, db
    
    def get_config(self):
        config = {
            "input_shape": self.input_shape,
            "filters": self.filter_count,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding
        }
        return config