import numpy as np


class Layer:
    def __init__(self, input_size, output_size, activation_function="relu", W=None, b=None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_funct = activation_function

        self.W, self.b = self._init_params()

        if W is not None:
            self.W = W
        if b is not None:
            self.b = b

    def _init_params(self):
        W = np.random.randn(self.output_size, self.input_size) * 0.01
        b = np.zeros((self.output_size, 1))
        return W, b

    def activation_function(self, x):
        if self.activation_funct == "relu":
            return np.maximum(0, x)
        if self.activation_funct == "sigmoid":
            return 1 / (1 + np.exp(-x))
        if self.activation_funct == "linear":
            return x

        raise ValueError(f"Unsupported activation function: {self.activation_funct}")

    def activation_function_derivative(self, x):
        if self.activation_funct == "relu":
            return np.where(x > 0, 1, 0)
        if self.activation_funct == "sigmoid":
            return x * (1 - x)
        if self.activation_funct == "linear":
            return np.ones_like(x)

        raise ValueError(f"Unsupported activation function: {self.activation_funct}")

    def forward(self, inputs):
        Z = np.dot(self.W, inputs) + self.b
        A = self.activation_function(Z)
        return Z, A