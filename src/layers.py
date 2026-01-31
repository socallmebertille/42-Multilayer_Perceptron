import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size, activation):
        self.W = np.random.randn(input_size, output_size)
        self.b = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, X):
        self.input = X
        self.Z = X @ self.W + self.b
        self.A = activate(self.Z, self.activation)
        return self.A

    def backward(self, dA, lr):
        dZ = dA * activate_derivative(self.Z, self.activation)
        dW = self.input.T @ dZ
        db = np.sum(dZ, axis=0, keepdims=True)

        self.W -= lr * dW
        self.b -= lr * db

        return dZ @ self.W.T

