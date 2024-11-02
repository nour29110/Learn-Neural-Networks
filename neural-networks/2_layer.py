# Coding a Layer

# Define a set of inputs, representing data passed to this layer of neurons
inputs = [1, 2, 3, 2.5]

# Define weights for each neuron in the layer; each list corresponds to one neuron's weights
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

# Define biases for each neuron, each value added to the respective neuron's weighted sum
bias1 = 2
bias2 = 3
bias3 = 0.5

# Calculate the output for each neuron in the layer by computing weighted sums of inputs + biases
output = [
    inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
    inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
    inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3
]

# Print the layer output, which contains the results from each neuron in the layer
print(output)
