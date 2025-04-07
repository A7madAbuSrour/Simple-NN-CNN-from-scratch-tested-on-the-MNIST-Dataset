import numpy as np

class Layer:
  def __init__(self):
    self.input = None
    self.output = None

  def forward(self, input): #returns output
    pass

  def backward(self, output_gradient, learning_rate): #update paramaters and return input gradient
    pass

class Dense(Layer):
  def __init__(self, input_size, output_size):
    self.weights = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
    self.bias = np.random.randn(output_size, 1)

  def forward(self, input):
    self.input = input
    return (self.weights @ self.input) + self.bias

  def backward(self, output_gradient, learning_rate):
    weights_gradient = (output_gradient @ self.input.T)
    self.weights -= learning_rate * weights_gradient
    self.bias -= learning_rate * output_gradient

    return (self.weights.T @ output_gradient)

class SoftMax(Layer):
  def forward(self, input):
    temp = np.exp(input)
    self.output = temp / np.sum(temp)
    return self.output

  def backward(self, output_gradient, learning_rate):
    n = np.size(self.output)
    temp = np.tile(self.output, n)
    return ((temp * (np.identity(n) - np.transpose(temp))) @ output_gradient)

class Activation(Layer):
  def __init__(self, activation, activation_prime):
    self.activation = activation
    self.activation_prime = activation_prime

  def forward(self, input):
    self.input = input
    return self.activation(self.input)

  def backward(self, output_gradient, learning_rate):
    return np.multiply(output_gradient, self.activation_prime(self.input))

class Tanh(Activation):
  def __init__(self):
    tanh = lambda x: np.tanh(x)
    tanh_prime = lambda x: 1-tanh(x)**2
    super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
  def __init__(self):
    sig = lambda x: 1 / (1+np.exp(-x))
    sig_prime = lambda x: sig(x) * (1-sig(x))
    super().__init__(sig, sig_prime)

def mse(y_true, y_pred): #mean sqaure function
  return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
  return 2 * (y_pred - y_true) / np.size(y_true)

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
  Dense(2, 3),
  Tanh(),
  Dense(3, 1),
  Sigmoid()
]

epochs = 1000
learning_rate = 0.1

for e in range(epochs):
  error = 0
  for x, y in zip(X, Y):
    #forward
    output = x
    for layer in network:
      output = layer.forward(output)

    #error
    error += mse(y, output)

    #backward
    grad = mse_prime(y, output)
    for layer in reversed(network):
      grad = layer.backward(grad, learning_rate)

  error /= len(X)
  print('%d/%d, error=%f' % (e + 1, epochs, error))

input = [[0],[0]]
for layer in network:
      input = layer.forward(input)

print(round(input[0][0]))