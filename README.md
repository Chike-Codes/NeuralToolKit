# NeuralToolKit (NTK)
A lightweight neural network framework built from scratch in Python using NumPy.

NeuralToolkit was designed to explore and implement the core mechanics of modern machine learning systems — including forward propagation, backpropagation, optimization algorithms, convolutional architectures, and reinforcement learning models.

This project focuses on understanding machine learning at a systems level rather than relying on high-level frameworks.

# Features

Modular neural network architecture

Manual backpropagation implementation

Convolutional Neural Network (CNN) support

Optimization algorithms:

Stochastic Gradient Descent (SGD)

Adam

RMSProp

Adagrad

# In Progress

Double Deep Q-Network (DDQN) reinforcement learning (Being Refactored)

NumPy-backed tensor abstraction

Designed for extensibility

# Installation
```powershell
python -m pip install -e neuraltoolkit
```

# XOR Example
```python
import neuraltoolkit as ntk
import numpy as np

data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

labels = np.array([
    [0],
    [1],
    [1],
    [0],

])

model = ntk.Model()
model.add_layer(ntk.Dense(activation=ntk.Relu(), 
                          input_shape=2, 
                          output_shape=5, 
                          initializer=ntk.He_initializer())
                          )
model.add_layer(ntk.Dense(activation=ntk.Sigmoid(), 
                          input_shape=5, 
                          output_shape=1, 
                          initializer=ntk.Glorot_initializer())
                          )
model.compile(optimizer=ntk.Adam(), loss=ntk.MeanSquaredError())

model.fit(x=data, y=labels, epochs=100, batch_size=4,)
```
