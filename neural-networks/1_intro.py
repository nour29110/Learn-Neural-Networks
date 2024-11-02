# Intro and Neuron Code

# Define the inputs to the neuron, representing values from previous layer or data points
inputs = [1.2, 5.1, 2.11]

# Define the weights, which represent the strength or importance of each input
weights = [3.1, 2.1, 8.71]

# Define the bias, a constant added to the neuron's weighted sum to shift its output
bias = 3

# Calculate the neuron's output as a weighted sum of inputs plus the bias
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias

# Print the result, showing the neuron's output based on the current inputs and weights
print(output)