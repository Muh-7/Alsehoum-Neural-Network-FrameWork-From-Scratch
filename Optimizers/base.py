# optimizers/base.py

from abc import ABC, abstractmethod


class Optimizer(ABC):

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    @abstractmethod
    def update(self, params, grads):
        pass

