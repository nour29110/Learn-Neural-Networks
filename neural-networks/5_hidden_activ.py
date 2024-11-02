# Hidden Layer Activation Function

import numpy as np
import nnfs
from nnfs.datasets import spiral_data 

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Define a class for the ReLU (Rectified Linear Unit) activation function
class Activation_ReLU:
    # Forward pass through the ReLU activation function
    def forward(self, inputs):
        # Apply the ReLU function: replace negative values with 0
        self.output = np.maximum(0, inputs)

# Create the first layer with 4 inputs and 5 neurons
layer1 = Layer_Dense(4, 5)

# Create a ReLU activation function instance for the first layer
activation1 = Activation_ReLU()

# Perform a forward pass through the first layer with input data
layer1.forward(X)

# Pass the output of the first layer through the ReLU activation function
activation1.forward(layer1.output)

# Print the output after applying ReLU, representing non-linear activation
print(activation1.output)
  