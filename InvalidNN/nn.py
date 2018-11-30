from InvalidNN.core import *
import tensorflow as tf
import multipledispatch


ACTIVATE = {
    'relu': tf.nn.relu
}


class Dense(Node):
    def __init__(self, name, scope, activate, units):
        super().__init__(name, scope)
        self.activate = ACTIVATE[activate]
        self.units = units
        self.weight = None
        self.bias = None

    def func(self, k):
        return self.activate(
            tf.matmul(self.weight) + self.bias
        )  # TODO: Pytorch


class NeuralNetwork(Graph):
    @abstractmethod
    def init_layers(self, layers, input_shape):
        pass  # Initialize Layers

    @abstractmethod
    def train(self, optimize, loss_fn, batch_size, epoch, learning_rate,
              train_data_generator, test_data_generator):
        pass  # Train Network, TODO: Normalization Method 추가(Batch Norm 등)

    @abstractmethod
    def accuracy(self, test_data_generator):
        pass  # Test Network, TODO: 자유도 높게

    @abstractmethod
    def __call__(self, input_data):
        pass  # Query To Network


'''
class TFNeuralNetwork(Graph):
    def __init__(self, name, scope, nodes, input_shape):
        super().__init__(nodes, name, scope)
        self.input_shape = input_shape

    def init_layers(self, layers, input_shape):
        pass
'''
