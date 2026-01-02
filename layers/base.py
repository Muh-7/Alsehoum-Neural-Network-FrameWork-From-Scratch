# layers/base.py

from abc import ABC, abstractmethod


class Layer(ABC):
    """
    Base class for all layers in the neural network.
    Every layer must implement forward and backward methods.
    """

    def __init__(self):
        self.params = {}
        self.grads = {}

    @abstractmethod
    def forward(self, x):
        
        pass

    @abstractmethod
    def backward(self, dout):
        
        pass

