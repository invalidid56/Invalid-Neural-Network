# Package Import
import tensorflow as tf

# Module Import
from multipledispatch import dispatch
from abc import *

# Init Layers


class Node(object):
    def __init__(self, name, connected: list=None):
        self.name: str = name
        if not connected:
            self.connected_node: list = connected[:-1]
            self.gathering: str = connected[-1]  # TODO: 커스텀 자유도 높이기
        else:
            self.connected_node = []  # fully-connected to previous nodes
            self.gathering = 'sum'


class Layer(Node):
    def __init__(self, name: str, activate: function, dropout: bool = False, connected: list=None):
        super().__init__(name=name, connected=connected)
        self.name = name
        self.activate = activate
        self.dropout = dropout

        self.weight = None  # TODO: 생성 메서드 커스텀 추가
        self.bias = None

    def __str__(self):
        pass


class Dense(Layer):
    category = 'Dense'

    def __init__(self, name, activate, units, dropout=False, connected=None):
        super().__init__(name=name, activate=activate, dropout=dropout, connected=connected)  # TODO: str 활성화함수 입력도 함수로 변환
        self.units = units


class Shortcut(Node):  # TODO: 감이 안잡힌당께
    flows = {}

    def __init__(self, name):
        super().__init__(name=name)


class WrongConnection(Exception):
    pass


class NeuralNetwork(metaclass=ABCMeta):
    def __init__(self, layers, input_shape):
        pass

    @abstractmethod
    def init_layers(self, layers, input_shape):
        pass

    @abstractmethod
    def query(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def accuracy(self):
        pass

    def __len__(self):
        pass

    def __str__(self):
        pass


class TFNetwork(NeuralNetwork):
    def init_layers(self, layers, input_shape):
        @dispatch()
        def init(layer, in_flow):
            pass

        @dispatch(Dense)
        def init(layer, in_flow):
            # TODO: 초기화 메서드에 자유를! 자비에 말고도 사용자 지정을 받아오게 하자, 아예 각각 메서드들을 따로 모듈화해서 구현하고 이를 사용자가 직접 오버라이딩 할 수 있게 해도
            layer.weight = tf.truncated_normal(

            )
            layer.bias = tf.truncated_normal(

            )
            if isinstance(in_flow, list):
                if layer.gathering == 'sum':
                    for flow in in_flow:

                result = None # TODO: Sum inputs
            else:
                result = layer.activate(
                    tf.matmul(in_flow, layer.weight) + layer.bias
                )
            return result


class PTNetwork(NeuralNetwork):
    def init_layers(self, layers, input_shape):
        pass

