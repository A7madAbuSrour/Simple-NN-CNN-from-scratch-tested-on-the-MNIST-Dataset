import numpy as np
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils

from SimpleNN import Layer, Sigmoid, Tanh, Dense
from scipy import signal

class Convolutional(Layer):
  def __init__(self, input_shape, kernel_size, depth):
    input_depth, input_height, input_width = input_shape
    self.depth = depth #output depth and number of kernels
    self.input_shape = input_shape
    self.input_depth = input_depth
    self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
    self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
    self.kernels = np.random.randn(*self.kernels_shape)
    self.biases = np.random.randn(*self.output_shape)

  def forward(self, input):
    self.input = input
    self.output = np.copy(self.biases)
    for i in range(self.input_depth):
      for j in range(self.input_depth):
        self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")

    return self.output

  def backward(self, output_gradient, learning_rate):
    kernels_gradient = np.zeros(self.kernels_shape)
    input_gradient = np.zeros(self.input_shape)

    for i in range(self.depth):
      for j in range(self.input_depth):
        kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
        input_gradient += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

    self.kernels -= learning_rate * kernels_gradient
    self.biases -= learning_rate * output_gradient

    return input_gradient

class Reshape(Layer):
  def __init__(self, input_shape, output_shape):
    self.input_shape = input_shape
    self.output_shape = output_shape

  def forward(self, input):
    return np.reshape(input, self.output_shape)

  def backward(self, output_gradient, learning_rate):
    return np.reshape(output_gradient, self.input_shape)

def binary_cross_entropy(y_true, y_pred):
  return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
  return ((1 - y_true) / (1 - y_pred) - (y_true / y_pred)) / np.size(y_true)

def data_prep_process(x, y, limit):
  zero_index = np.where(y == 0)[0][:limit]
  one_index = np.where(y == 1)[0][:limit]

  total_indices = np.hstack((zero_index, one_index))
  total_indices = np.random.permutation((total_indices))
  x, y = x[total_indices], y[total_indices]

  x = np.reshape(x, (len(x), 1, 28, 28))
  x = x.astype("float32") / 255

  y = utils.to_categorical(y, 2)
  y = y.reshape(len(y), 2, 1)
  return x, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = data_prep_process(x_train, y_train, 100)
x_test, y_test = data_prep_process(x_test, y_test, 100)

# neural network
network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
]

epochs = 20
learning_rate = 0.01

#train
for e in range(epochs):
  error = 0
  for x, y in zip(x_train, y_train):
    #forwarding
    output = x
    for layer in network:
      output = layer.forward(output)

    error += binary_cross_entropy(y, output)

    #backwarding
    gradient = binary_cross_entropy_prime(y, output)
    for layer in reversed(network):
      gradient = layer.backward(gradient, learning_rate)

  error /= len(x_train)
  print(f"{e + 1}/{epochs}, error={error}")

for x,y in zip(x_test, y_test):
  output = x
  for layer in network:
    output = layer.forward(output)
  print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")