# layers/dense.py

import numpy as np
from layers.base import Layer


class Dense(Layer):
    """
    (affine) layer:
    y = xW + b
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        #Hon Nehna AM Initialize weights and bias
        self.params["W"] = 0.01 * np.random.randn(input_dim, output_dim)
        self.params["b"] = np.zeros(output_dim)

        self.x = None  # cache input for backward pass

    def forward(self, x):
    
        self.x = x
        W = self.params["W"]
        b = self.params["b"]

        out = np.dot(x, W) + b
        return out

    def backward(self, dout):
        
        W = self.params["W"]

        # Gradients
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        # Store gradients
        self.grads["W"] = dW
        self.grads["b"] = db

        return dx

