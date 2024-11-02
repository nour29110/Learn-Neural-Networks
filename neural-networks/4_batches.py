# Batches, Layers, and Objects

import numpy as np

# Set a random seed for reproducibility, ensuring the same random values each time the code runs
np.random.seed(0)

# Define a batch of input data, with each sublist representing an individual input sample
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# Define a class for a dense (fully connected) layer in the neural network
class Layer_Dense:
    # Initialize the layer with the number of inputs and neurons
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with random values, scaled by 0.10 for smaller initial values
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # Initialize biases as zeros, one for each neuron
        self.biases = np.zeros((1, n_neurons))

    # Define the forward pass, computing the layer output
    def forward(self, inputs):
        # Calculate the dot product of inputs and weights, then add biases
        self.output = np.dot(inputs, self.weights) + self.biases  

# Create two layers: the first with 4 inputs and 5 neurons, the second with 5 inputs and 2 neurons
layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

# Perform a forward pass through the first layer using the input data X
layer1.forward(X)

# Perform a forward pass through the second layer, using the output of the first layer as its input
layer2.forward(layer1.output)

# Print the output of the second layer, representing the final result of this network
print(layer2.output)

