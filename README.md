# Learn Deep Learning, Step by Step Guide

This repository is a beginner friendly, step by step tutorial for understanding Deep Learning from scratch. Each Python file introduces a new concept, gradually building up from the basics of neurons to complex techniques. Start off from neural-networks folder, `1_intro.py` to `9_optimization.py` to start learning about Neural Networks from scratch. Then follow along to learn about Recurrent Neural Networks (RNN), Gated Recurrent Units (GRU), and Long Short-Term Memory (LSTM) networks, using the PyTorch framework.

---

## Table of Contents
1. [Learn_Deep_Learning/neural-networks/1_intro.py - Introduction and Neuron Code](#1-introduction-and-neuron-code)
2. [neural-networks/2_layer.py - Coding a Layer](#2-coding-a-layer)
3. [neural-networks/3_dot_product.py - Dot Product in Neural Networks](#3-dot-product-in-neural-networks)
4. [neural-networks/4_batches.py - Batches, Layers, and Objects](#4-batches-layers-and-objects)
5. [neural-networks/5_hidden_activ.py - Hidden Layer Activation Function (ReLU)](#5-hidden-layer-activation-function-relu)
6. [neural-networks/6_softmax_activ.py - Softmax Activation Function](#6-softmax-activation-function)
7. [neural-networks/7_loss_categ.py - Calculating Loss with Categorical Cross-Entropy](#7-calculating-loss-with-categorical-cross-entropy)
8. [neural-networks/8_implementing_loss.py - Implementing Categorical Cross-Entropy Loss in Classes](#8-implementing-categorical-cross-entropy-loss-in-classes)
9. [neural-networks/RNN/rnn.py - RNN Implementation](#9-rnn-implementation)
10. [Learn_Deep_Learning/RNN/utils.py - RNN Utilities](#10-rnn-utilities)
11. [Learn_Deep_Learning/RNN/rnn-lstm-gru/main.py - Main Program](#11-main-program)


---

## 1. Introduction and Neuron Code
**File:** `1_intro.py`  
**Concepts:** Introduces the concept of a single neuron. In this script, we:
- Define input values, weights, and a bias.
- Calculate the output of a simple neuron using these values.
This file establishes the basic building blocks of neural networks.

---

## 2. Coding a Layer
**File:** `2_layer.py`  
**Concepts:** Expands on `1_intro.py` by creating a layer of multiple neurons.
- Each neuron has its own set of weights and biases.
- The output for each neuron in the layer is computed individually.
This file shows how to scale the neuron concept to create a layer.

---

## 3. Dot Product in Neural Networks
**File:** `3_dot_product.py`  
**Concepts:** Uses matrix multiplication (dot product) to compute outputs of multiple neurons at once.
- By using NumPy’s dot product, we simplify layer computations.
- This approach reduces repetitive code and improves efficiency.
This file demonstrates the mathematical power behind neural network computations.

---

## 4. Batches, Layers, and Objects
**File:** `4_batches.py`  
**Concepts:** Introduces the `Layer_Dense` class to structure layers and supports batch inputs.
- We define a class with methods for weights, biases, and forward passes.
- This file also handles batches of inputs for a more scalable approach to neural networks.

---

## 5. Hidden Layer Activation Function (ReLU)
**File:** `5_hidden_activ.py`  
**Concepts:** Introduces the **ReLU (Rectified Linear Unit)** activation function.
- This function helps the model learn complex patterns by applying non-linearity to the outputs.
- It prepares for deeper layers where non-linear activation functions are essential for capturing data complexity.

---

## 6. Softmax Activation Function
**File:** `6_softmax_activ.py`  
**Concepts:** Introduces the **Softmax** activation function, commonly used in output layers for classification tasks.
- The softmax function converts outputs into probabilities, useful for multi-class classification.
- This file shows how to calculate probabilities for each class, making the network’s predictions interpretable.

---

## 7. Calculating Loss with Categorical Cross-Entropy
**File:** `7_loss_categ.py`  
**Concepts:** Introduces the concept of loss and the **Categorical Cross-Entropy** loss function.
- Loss functions measure the accuracy of the model’s predictions.
- This file provides a basic example of using cross-entropy to quantify prediction error, guiding the model’s learning process.

---

## 8. Implementing Categorical Cross-Entropy Loss in Classes
**File:** `8_implementing_loss.py`  
**Concepts:** Builds on `7_loss_categ.py` by implementing the cross-entropy loss function within a class.
- This structure allows for easy reusability and integration with larger projects.
- We calculate the average loss across multiple samples, a key step in training neural networks.

---

## 9. Optimization and Derivatives
**File:** `9_optimization.py`  
**Concepts:** Explores **optimization** and **derivatives** as they relate to gradient descent.
- Introduces derivatives as a way to understand how changing inputs affects outputs.
- We plot tangent lines and approximate derivatives, which are essential for backpropagation in neural networks.

---
---
# Contents
- **main.py**: Implements an RNN to classify handwritten digits from the MNIST dataset.
- **rnn.py**: Implements a character-level RNN for name classification based on the language of origin.
- **utils.py**: Contains utility functions for loading data and processing inputs.

## Recurrent Neural Networks (RNN)
RNNs are a class of neural networks designed for sequence data, where the output from the previous step is fed as input to the current step. Unlike traditional feedforward networks, RNNs can maintain a hidden state that gets updated at each time step, allowing them to learn from sequences of arbitrary length.

### Key Components:
- **Hidden State**: The memory of the network that is passed from one time step to the next. This allows the network to capture information about previous inputs.
- **Input Sequence**: The input to the RNN is typically a sequence of vectors. In `main.py`, each image from the MNIST dataset is represented as a sequence of pixels (28 pixels per row over 28 rows).

## Gated Recurrent Units (GRU) and Long Short-Term Memory (LSTM)
Both GRU and LSTM are advanced types of RNNs designed to address the vanishing gradient problem that standard RNNs face when learning long-range dependencies in sequences.

### LSTM:
It introduces three gates:
- **Input Gate**: Controls how much of the new information to let into the memory.
- **Forget Gate**: Decides how much of the old memory to forget.
- **Output Gate**: Determines how much of the memory to output to the next time step.

This architecture allows LSTMs to retain information over long sequences, making them effective for tasks such as language modeling and translation.

### GRU:
It combines the forget and input gates into a single update gate. This results in a simpler architecture while still effectively capturing dependencies over time.

## Implementation Details:

## main.py
In `main.py`, we implement a simple RNN to classify the MNIST digits:

### Dataset Loading:
- The MNIST dataset is loaded using torchvision, which contains images of handwritten digits (0-9).
- Each image is transformed into a tensor and loaded into batches using DataLoader.

### Model Definition:
- The RNN class defines the network structure: an RNN layer followed by a fully connected (linear) layer to produce class probabilities.
- The forward method processes the input sequence and returns the output.

### Training Loop:
- The model is trained over several epochs, calculating the loss using cross-entropy loss and updating the model weights via backpropagation.

### Testing Phase:
- After training, the model's accuracy is evaluated on a test dataset.

## rnn.py
In `rnn.py`, we define an RNN for classifying names based on their languages:

### Model Architecture:
- The RNN class includes input-to-hidden (i2h) and input-to-output (i2o) layers, along with softmax activation to get probabilities for each category.

### Training Process:
- The training function iterates over the characters in a name, updating the hidden state at each step and calculating loss using Negative Log Likelihood Loss.

### Prediction:
- A separate function is provided to predict the category of a name based on the trained model.

## utils.py
This module provides various utility functions for data preprocessing:

- **Data Loading**: Loads names from text files categorized by language, normalizes the names to ASCII, and prepares the data for training.
- **Tensor Conversion**: Converts letters to one-hot encoded tensors for input to the model.
- 
---

## How to Run the Code
- **Neural Networks folder:** To run these files, you’ll need: - Python (preferably version 3.7 or later) - Required libraries: `numpy`, `matplotlib`, and `nnfs` Install libraries using:
```bash
  pip3 install numpy matplotlib nnfs
```
- **Install Dependencies for RNN folder:** Make sure you have PyTorch installed along with torchvision.
```bash
  pip3 install pytorch
```
- **Download Data:** Run the code to download the MNIST dataset automatically. For name classification, ensure that the dataset of names is available in the `data/names` directory.
- **Run the Scripts:**
  - For digit classification, run `python main.py`.
  - For name classification, run `python rnn.py`.

---

## Conclusion
This repository provides a foundational understanding of Neural Networds, RNNs, GRUs, and LSTMs through practical implementations in python libraries and frameworks. 
