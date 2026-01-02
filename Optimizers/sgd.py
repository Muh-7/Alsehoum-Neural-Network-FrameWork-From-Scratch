# optimizers/sgd.py

from Optimizers.base import Optimizer


class SGD(Optimizer):
    

    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]

