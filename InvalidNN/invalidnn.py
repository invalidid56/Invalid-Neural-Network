# Package Import
import tensorflow as tf

# Module Import
from multipledispatch import dispatch
from abc import *

# Init Layers


class Node(object):
    def __init__(self, name, gathering=None):
        self.name = name
        if gathering:
            if isinstance(gathering, tuple):
                self.connected = gathering[:-1]
                self.gathering = gathering[-1]
            else:
                self.connected = None
                self.gathering = gathering
        else:
            self.gathering = None


class Layer(Node):
    def __init__(self, name: str, activate, dropout: bool = False, gathering=None):
        super().__init__(name=name, gathering=gathering)
        self.name = name
        self.activate = activate
        self.dropout = dropout

        self.weight = None  # TODO: 생성 메서드 커스텀 추가, 프로퍼티?
        self.bias = None

    def __str__(self):
        pass


class Dense(Layer):
    category = 'Dense'

    def __init__(self, name, activate, units, dropout=False, gathering=None):
        super().__init__(name=name, activate=activate, dropout=dropout, gathering=gathering)  # TODO: str 활성화함수 입력도 함수로 변환
        self.units = units


class Shortcut(Node):  # TODO: 감이 안잡힌당께
    flows = {}

    def __init__(self, name):
        super().__init__(name=name)


class WrongConnection(Exception):
    pass


class NeuralNetwork(metaclass=ABCMeta):
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


class MixinNN(object):
    def __len__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class TFNetwork(NeuralNetwork, MixinNN):
    def __init__(self, layers, input_shape):
        self.flow = self.init_layers(layers, input_shape)

    def init_layers(self, layers, input_shape):
        @dispatch(Dense, object)
        def init(layer, in_flow):
            #
            if layer.gathering == 'concat':
                input = tf.concat(1, in_flow)
            elif layer.gathering == 'sum':
                temp_flow = in_flow
                input = in_flow[0]
            else:
                input = in_flow

            #
            layer.weight = tf.truncated_normal(shape=[
                input.get_shape().as_list()[-1], layer.units
            ]) / tf.sqrt(float(layer.units))
            layer.bias = tf.truncated_normal(shape=[layer.units])

            #
            activate = lambda x: layer.activate(
                tf.matmul(input, layer.weight) + layer.bias
            )
            return activate(input) if not layer.gathering == 'sum' else sum([activate(i) for i in temp_flow])

        flow = tf.placeholder(tf.float32, input_shape)
        for layer in layers:
            if isinstance(layer, list):
                flow = [init(node, flow) for node in layer]
            else:
                flow = init(layer, flow)
        return flow

    def query(self):
        pass

    def train(self):
        pass

    def accuracy(self):
        pass


'''
class TFNetwork(NeuralNetwork):
    def init_layers(self, layers, input_shape):
        @dispatch()
        def init(layer, in_flow):
            pass

        @dispatch(Dense)
        def init(layer: Dense, in_flow):
            # TODO: 초기화 메서드에 자유도 자비에 말고도 사용자 지정을 받아오게 하자, 아예 각각 메서드들을 따로 모듈화해서 구현하고 이를 사용자가 직접 오버라이딩 할 수 있게 해도

            if layer.gathering == 'concat':
                in_flow = tf.concat(1, in_flow)
            elif layer.gathering == 'sum':
                temp = in_flow
                in_flow = in_flow[0]
            layer.weight = tf.truncated_normal(shape=[
                        in_flow.get_shape().as_list()[-1], layer.units
                    ])/tf.sqrt(layer.units)
            layer.bias = tf.truncated_normal(shape=[layer.units])
            if not layer.gathering == 'sum':
                result = layer.activate(
                    tf.matmul(in_flow, layer.weight) + layer.bias
                )
            else:
                result = sum([layer.activate(
                    tf.matmul(i_f, layer.weight) + layer.bias
                ) for i_f in temp])
            return result
'''


class PTNetwork(NeuralNetwork):
    def init_layers(self, layers, input_shape):
        pass


# TODO: 리커런트?

