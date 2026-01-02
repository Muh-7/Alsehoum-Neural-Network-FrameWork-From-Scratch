# optimizers/adam.py
import numpy as np
from Optimizers.base import Optimizer


class Adam(Optimizer):
    

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        self.t += 1

        for key in params:
            param = params[key]
            grad = grads[key]

            pid = id(param)  # unique id

            if pid not in self.m:
                self.m[pid] = np.zeros_like(param)
                self.v[pid] = np.zeros_like(param)

            self.m[pid] = self.beta1 * self.m[pid] + (1 - self.beta1) * grad
            self.v[pid] = self.beta2 * self.v[pid] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[pid] / (1 - self.beta1 ** self.t)
            v_hat = self.v[pid] / (1 - self.beta2 ** self.t)

            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
