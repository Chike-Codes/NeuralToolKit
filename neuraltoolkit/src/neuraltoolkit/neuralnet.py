import numpy as np
from . import losses
from . import layers
from . import data
from .optimizers.optimizer import Optimizer
from .graph import Graph
import pickle
from matplotlib import pyplot as plt
from copy import deepcopy

class Model:
    def __init__(self):
        self.layers = []
        self.layer_count = 0

    def add_layer(self, layer):
        

        self.layers.append(layer)
        self.layer_count += 1

    def compile(self, optimizer:Optimizer, loss:object):
        input_shapes = [self.layers[0].get_inputDim()]

        self.loss = loss

        for i, layer in enumerate(self.layers):
            if isinstance(layer, layers.Concatenate):
                input_shapes.append(layer.get_concatDim())
            if layer.input_shape == None:
                if i == 0:
                    raise ValueError("No input shape assigned.")
                else:
                    layer.input_shape = self.layers[i-1].output_shape
            
            layer.initialize_parameters()
            layer.initialize_optimizers(deepcopy(optimizer))

        self.model_inputDim = tuple(input_shapes)

    def feed_forward(self, input_data:np.array, aux_input=None):
        activations = []
        activation_derivatives = []
        for i in range(self.layer_count):
            input = input_data if i == 0 else activations[i - 1]
            if isinstance(self.layers[i], layers.Concatenate):
                input = np.concatenate((input, aux_input), axis=1)
                aux_idx += 1
            layer_activations, layer_activation_derivatives = self.layers[i].forward(input)

            activations.append(layer_activations)
            activation_derivatives.append(layer_activation_derivatives)
        return activations, activation_derivatives
    
    def backprop(self, activations:list[list[float]], activation_derivatives:list[list[float]], input_data:list, output_data:list, batch_size:int):
        backprop_error = self.loss.derive(activations[:][-1], output_data, batch_size)
        for i in reversed(range(self.layer_count)):
            input = input_data if i == 0 else activations[i - 1]
            if i != self.layer_count - 1:
                backprop_error = self.layers[i + 1].back(error=backprop_error, derivatives=activation_derivatives[i])
                weight_gradient, bias_gradient = self.layers[i].calc_gradients(error=backprop_error, input=input)
                self.layers[i].optimize_parameters(weight_gradient, bias_gradient)
            else:
                weight_gradient, bias_gradient = self.layers[i].calc_gradients(error=backprop_error, input=input)
                self.layers[i].optimize_parameters(weight_gradient, bias_gradient)
            
    def predict(self, input:list[list], aux_input=None):
        outputs = self.feed_forward(input, aux_input)[0]
        return outputs[-1]
    
    def validation_loss(self, test_x, test_y, epoch):
        batch_count = len(test_x)
        loss = np.zeros(batch_count)
        for i in range(batch_count):
            output = self.predict(test_x[i])
            loss[i] = np.sum(self.loss.loss(output, test_y[i]))
            print(f"Epoch: {epoch} - Validation Batch: {i} / {batch_count}", end="\r")
        loss = np.mean(loss)
        return loss
    
    def fit(
            self, 
            x=None, 
            y=None, 
            aux=None,
            epochs:int=1, 
            batch_size:int=32, 
            shuffle:bool=True, 
            validation_data:tuple=None,
            validation_split=0.0,
            validation_rate=1,
            validation_batch_size=32,
            use_graph:bool=False, 
            graph_rate:int=1,
    ):
        sample_count = len(x)

        v_loss = 0
        validation_data = data.split_validation_data(validation_data, validation_split, validation_batch_size)
        
        if use_graph:
            graph = Graph("Epochs", "Loss", "Loss Over Time")
            graph.add_plot("Loss")
        
        for epoch in range(epochs):
            training_x = x.copy
            training_y = y.copy
            epoch_loss = 0
            if shuffle:
                training_x, training_y = data.shuffle_data(x, y, np.random.randint(0, 10000))
            training_x = data.create_batches(training_x, batch_size)
            training_y = data.create_batches(training_y, batch_size)
            batches = sample_count // batch_size

            for batch in range(batches):
                epoch_loss = self.fit_batch(training_x[batch], training_y[batch])
                print(f"Epoch: {epoch} - Batch: {batch} / {batches}", end="\r")
            epoch_loss = np.sum(epoch_loss)

            if validation_data != None and epoch % validation_rate == 0:
                validation_loss = self.validation_loss(validation_data[0], validation_data[1], epoch)
                print(f"Epoch: {epoch} - Loss: {epoch_loss} - Validation Loss: {round(validation_loss, 4)}", end="\n")
                v_loss = validation_loss
            else:
                print(f"Epoch: {epoch} - Loss: {epoch_loss}", end="\n")            
            
            if epoch % graph_rate == 0 and use_graph:
                graph.append("Loss", epoch, epoch_loss)
                graph.update()
    
    def fit_batch(self, training_x, training_y, aux_input=None):
        sample_count = len(training_x)
        activations, activation_derivatives = self.feed_forward(training_x, aux_input)
        self.backprop(activations, activation_derivatives, training_x, training_y, sample_count)
        return self.loss.loss(activations[:][-1], training_y)
    
    def is_valid_input(self, x):
        if x.shape[1:] == self.layers[0].input_shape:
            return True
        else:
            return False
    
    def save(self, filepath:str="/model_struct"):
        model_data = {
            "layers": [layer for layer in self.layers],
            "loss": self.loss,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print("Model Saved")
    
    def load(self, filepath:str):
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        for layer_data in model_data["layers"]:
            layer = deepcopy(layer_data)
            self.layers.append(layer)
            self.layer_count += 1
        self.loss = model_data["loss"]

    def logits(self, x):
        activations = []
        logits = []
        for i in range(self.layer_count):
            input = x if i == 0 else activations[-1]
            logits.append(self.layers[i].logits(input))
            activations.append(self.layers[i].forwardDerive(input)[0])
        return logits