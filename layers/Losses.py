# layers/losses.py

import numpy as np
from layers.base import Layer


# HOOn Hessab al mean sequare Error.
class MeanSquaredError(Layer):
  

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        loss = 0.5 * np.mean((y_pred - y_true) ** 2)
        return loss

    def backward(self):
        batch_size = self.y_true.shape[0]
        dx = (self.y_pred - self.y_true) / batch_size
        return dx


class SoftmaxCrossEntropy(Layer):


    def forward(self, scores, y_true):
        self.y_true = y_true

        # Numerical stability
        scores = scores - np.max(scores, axis=1, keepdims=True)

        exp_scores = np.exp(scores)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        batch_size = scores.shape[0]
        correct_logprobs = -np.log(self.probs[range(batch_size), y_true])
        loss = np.sum(correct_logprobs) / batch_size

        return loss

    def backward(self):
        batch_size = self.y_true.shape[0]
        dx = self.probs.copy()
        dx[range(batch_size), self.y_true] -= 1
        dx /= batch_size
        return dx

