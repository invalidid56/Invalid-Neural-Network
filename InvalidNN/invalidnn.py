# Package Import

# Module Import
import math
from abc import *

# Init Layers


class Node(object):
    def __init__(self):
        pass


class CompressLayer(Node):
    def __init__(self):
        super().__init__()
        self.in_flows = None


class Sum(CompressLayer):
    pass


class Concat(CompressLayer):
    pass


class Layer(Node):
    def __init__(self, name: str, activate: str, dropout: bool):
        super().__init__()
        self.name = name
        self.activate = activate.upper()
        self.dropout = dropout

    def __str__(self):
        pass


class Dense(Layer):
    category = 'Dense'

    def __init__(self, name, activate, units, dropout=False):
        super().__init__(name, activate, dropout)
        self.units = units


class Shortcut(Layer):
    flows = {}
    pass


class WrongConnection(Exception):
    pass


class NeuralNetwork(object):
    def __init__(self, layers, input_shape):
        pass

    @abstractmethod
    def init_layers(self, layers, input_shape):
        pass

    def query(self):
        pass

    def train(self):
        pass

    def accuracy(self):
        pass

    def __len__(self):
        pass

    def __str__(self):
        pass


class TFNetwork(NeuralNetwork):
    def init_layers(self, layers, input_shape):
        pass


class PTNetwork(NeuralNetwork):
    def init_layers(self, layers, input_shape):
        pass

