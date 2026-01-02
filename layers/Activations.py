# layers/activations.py

import numpy as np
from layers.base import Layer


class Linear(Layer):


    def forward(self, x):
        self.x = x
        return x

    def backward(self, dout):
        return dout


class ReLU(Layer):


    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        dx = dout.copy()
        dx[self.x <= 0] = 0
        return dx


class Sigmoid(Layer):


    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx


class Tanh(Layer):


    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        dx = dout * (1 - self.out ** 2)
        return dx

