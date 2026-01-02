# optimizers/momentum.py

from Optimizers.base import Optimizer


class Momentum(Optimizer):


    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = 0

            self.velocity[key] = (
                self.momentum * self.velocity[key] - self.lr * grads[key]
            )
            params[key] += self.velocity[key]

