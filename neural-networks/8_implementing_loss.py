# Implementing Loss

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward (self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    # Base loss class to calculate the average loss over multiple samples
    def calculate(self, output, y):
        # Calculate sample losses using a specific forward method (to be defined in subclasses)
        sample_losses = self.forward(output, y)
        # Calculate mean loss across all samples for the batch
        data_loss = np.mean(sample_losses)
        return data_loss

# Specialized class for categorical cross-entropy loss calculation
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass to calculate categorical cross-entropy loss for predictions and true labels
    def forward(self, y_pred, y_true):
        # Number of samples in the batch
        samples = len(y_pred)
        # Clip predictions to avoid logarithm of zero errors, ensuring stable computations
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # If labels are given as class indices (1D array), select the correct prediction confidence
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # If labels are one-hot encoded (2D array), calculate confidence as dot product with predictions
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Compute negative log likelihoods for each sample
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

# Generate dataset
X, y = spiral_data(samples=100, classes=3)

# Create layers and activation functions
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# Forward pass through the first layer and activation
dense1.forward(X)
activation1.forward(dense1.output)

# Forward pass through the second layer and softmax activation
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])  # Output first 5 samples for verification

# Calculate loss
loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)
