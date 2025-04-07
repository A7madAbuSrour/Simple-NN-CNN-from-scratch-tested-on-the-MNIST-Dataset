# Simple Neural Network and CNN from Scratch

This repository contains a basic implementation of a feedforward Neural Network (NN) and a Convolutional Neural Network (CNN), fully built from scratch using only core Python — no machine learning libraries like TensorFlow or PyTorch were used.

---

### 🧠 Neural Network (`SimpleNN.py`)

The `SimpleNN.py` file defines the core building blocks for a simple neural network.

#### 📦 Base Class
- **`Layer`**  
  An abstract parent class that provides a template for all layers in the network.

#### 🌿 Child Classes of `Layer`
- **`Dense`**  
  A fully connected layer that applies a linear transformation (`Wx + b`) to its input.
  
- **`SoftMax`**  
  Applies the softmax activation function to convert raw output scores into probabilities.
  
- **`Activation`**  
  A generic activation layer that delegates to specific activation functions (e.g., Tanh, Sigmoid).

#### 🌱 Subclasses of `Activation`
- **`Tanh`**  
  Implements the tanh activation function for introducing non-linearity.
  
- **`Sigmoid`**  
  Implements the sigmoid activation function, often used in binary classification problems.

---

### 📷 Convolutional Neural Network (`SimpleCNN.py` (Testing network using MNIST dataset using class 1 and class 0) & `SimpleCNN_2_3.py` (Testing network using MNIST dataset using class 2 and class 3))

The `SimpleCNN.py` & `SimpleCNN_2_3.py` files build on top of `SimpleNN.py` by importing and reusing its layers.

#### 🌿 CNN-Specific Child Classes of `Layer`
- **`Convolutional`**  
  A simple convolutional layer that applies filters to input data, simulating spatial feature extraction.
  
- **`Reshape`**  
  A utility layer that reshapes 2D or multi-dimensional inputs into flat vectors for transition to fully connected layers.

---

### 🛠 Notes

- Each layer class (and its children) is implemented **entirely from scratch** using NumPy and basic Python.
- This project is intended for educational purposes, to demonstrate how the core components of NNs and CNNs function under the hood.
