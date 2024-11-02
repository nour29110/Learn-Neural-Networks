# Softmax Activation 

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward (self, inputs):
        self.output = np.maximum(0, inputs)

# Define a new activation function class for the Softmax function
class Activation_Softmax:
    # Forward pass for the Softmax activation
    def forward(self, inputs):
        # Subtracting the max input value per sample (row) for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Calculating normalized probabilities by dividing exponentiated values by their row-wise sum
        probabilties = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        # Storing the final probabilities as the output of this layer
        self.output = probabilties

# Generate a synthetic dataset for testing the network
X, y = spiral_data(samples=100, classes=3)

# Create layers with defined neurons, using ReLU and Softmax for activation
dense1 = Layer_Dense(2, 3)    # First dense layer with 2 inputs, 3 neurons
activation1 = Activation_ReLU()  # ReLU activation for the first layer

dense2 = Layer_Dense(3, 3)     # Second dense layer with 3 inputs, 3 neurons
activation2 = Activation_Softmax()  # Softmax activation for output layer

# Forward pass through the first layer and ReLU activation
dense1.forward(X)
activation1.forward(dense1.output)

# Forward pass through the second layer and Softmax activation
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Print the first five results to examine the output probabilities
print(activation2.output[:5])
