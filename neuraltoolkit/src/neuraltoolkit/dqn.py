from .neuralnet import Model
from .losses import MeanSquaredError
from .graph import Graph
from .replaybuffers import MultiReplayBuffer, ReplayBuffer, Replay
from . import ops
import numpy as np
import pickle
from collections import deque
from copy import deepcopy
import time
class Dqn:
    def __init__(self):
        self.policy = Model()
        self.cycle_count = 0
        self.error = 0

    def create_graph(self):
        self.graph = Graph("Batches", "Loss", "Loss Over Time")
        self.graph.add_plot("Loss", "b")

    def add_layer(self, layer):
        self.policy.add_layer(layer)
    
    def compile(self,
            optimizer, 
            epsilon_max:float=1,
            epsilon_min:float=0, 
            epsilon_decay:float=0.9999,
            discount_rate:float=0.99, 
            buffer=ReplayBuffer,
            buffer_count=1,
            buffer_split=None,
            buffer_cap=100_000, 
            buffer_min=1000, 
            n_step_cap=1,
            alpha=0.6,
            config:dict=None, 
            **kwargs
            ):
        loss = meanSquaredError
        self.policy.compile(optimizer, loss, config, **kwargs)
        self.target = deepcopy(self.policy)
        self.inputDim = self.policy.model_inputDim
        self.output_space = self.policy.layers[-1].output_shape

        self.discount_rate = discount_rate
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = self.epsilon_max

        self.replay_buffer = buffer(
            buffers=buffer_count,
            buffer_split=buffer_split,
            inputDim=self.inputDim,
            capacity=buffer_cap, 
            min=buffer_min, 
            n_step_cap=n_step_cap, 
            discount_rate=discount_rate,
            alpha=alpha
            )
        
    def fit(self, batch_size:int=32, beta=1):
        self.cycle_count += 1

        transitions, indices, IS_weights = self.replay_buffer.sample(batch_size=batch_size, beta=beta)

        states = transitions["states"]
        new_states = transitions["new_states"]
        actions = transitions["actions"]
        rewards = transitions["rewards"]
        flags = transitions["done_flags"]
        flag_indices = np.where(flags == True)[0]


        activations, activation_derivatives = self.policy.feed_forward(states)
        target_activations = self.target.feed_forward(new_states)[0]
        policy_predictions = np.array([pp[actions[i]] for i, pp in enumerate(activations[-1])])
        predicted_rewards = np.array([np.argmax(ta) for ta in target_activations[-1]])
        predicted_rewards[flag_indices] = 0
        target_predictions = rewards + self.discount_rate * predicted_rewards
        td_errors = policy_predictions - target_predictions
        loss_derivative = td_errors * (2 * IS_weights / batch_size)
        loss_derivative = loss_derivative[:, np.newaxis] * ops.oneHot(actions, self.output_space)

        self.policy.backprop_loss(loss_derivative, activations, activation_derivatives, states)

        self.error = np.sum(np.power(td_errors * IS_weights, 2)) / batch_size

        self.replay_buffer.update_td_errors(indices=indices, td_errors=td_errors)

    def graph_error(self):
        self.graph.append("Loss", self.cycle_count, self.error)
        self.graph.update()

    def update_target(self):
        self.target = deepcopy(self.policy)

    def predict(self, input):
        if not isinstance(input, (tuple, list)):
            input = (input,)
        output = self.policy.predict(input)
        return output
    
    def save(self, filepath:str="./model_struct"):
        model_data = {
            "policy": self.policy,
            "target": self.target,
            "output_space": self.output_space,
            "discount_rate": self.discount_rate,
            "replay_buffer": self.replay_buffer
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print("Model Saved")

    def load(self, filepath:str):
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.policy = model_data["policy"]
        self.target = model_data["target"]
        self.output_space = model_data["output_space"]
        self.discount_rate = model_data["discount_rate"]
        self.replay_buffer = model_data["replay_buffer"]
        print("Model Loaded")