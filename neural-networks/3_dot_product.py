# The Dot Product 

# Import the NumPy library for efficient mathematical operations
import numpy as np

# Define inputs, representing data passed into this layer of neurons
inputs = [1, 2, 3, 2.5]

# Define a weight matrix, where each sublist represents weights for a single neuron
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

# Define biases for each neuron, each added to the respective neuron's weighted sum
biases = [2, 3, 0.5]

# Use the dot product to calculate the weighted sum of inputs for each neuron
# Adding biases to the result to get the final output of the layer
output = np.dot(weights, inputs) + biases

# Print the output for the entire layer, showing the result of each neuron's activation
print(output)
