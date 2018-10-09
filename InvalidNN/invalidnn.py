# Import
import tensorflow as tf
import torch
from multipledispatch import dispatch
from abc import *
from random import choice

import math

# Node Class


class Node(object):
    @dispatch(str, object)
    def __init__(self, name, gathering):
        self.name = name
        self.gathering = None

    @dispatch(str, str)
    def __init__(self, name, gathering):
        self.name = name
        self.gathering = gathering
        self.connected = None

    @dispatch(str, tuple)
    def __init__(self, name, gathering):
        self.name = name
        self.gathering = gathering[0]
        self.connected = gathering[1:]


class Layer(Node):
    def __init__(self, name: str, activate, dropout: bool=False, gathering=None):
        super().__init__(name, gathering)
        self.activate = activate
        self.dropout = dropout

        self.weight = None  # TODO: Property 설정
        self.bias = None  # TODO: Property 설정

    def __str__(self):
        pass  # TODO: 속성 정리하여 출력하도록


class Dense(Layer):
    def __init__(self, name, activate, units, dropout=False, gathering=None):
        super().__init__(name=name, activate=activate, dropout=dropout, gathering=gathering)

        self.units = units


class Conv2D(Layer):
    def __init__(self, name, activate, filters, filter_size, stride, padding: str, dropout=False, gathering=None):
        super().__init__(name=name, activate=activate, dropout=dropout, gathering=gathering)

        self.filters = filters
        self.filter_size = filter_size
        self.stride = [1, *stride, 1]  # TODO: 예외처리-사이즈가 반복가 아닐 때
        self.padding = padding.upper()


class Pooling(Layer):
    def __init__(self, name, pooling, size, stride, padding: str, activate=None, dropout=False, gathering=None):
        super().__init__(name, activate, dropout, gathering)

        self.pooling = pooling
        self.size = [1, *size, 1]
        self.stride = [1, *stride, 1]
        self.padding = padding.upper()

# Network Class


class Network(metaclass=ABCMeta):
    @abstractmethod
    def init_layers(self, layers, input_shape):
        pass  # Initialize Layers

    @abstractmethod
    def train(self, optimize, loss_fn, batch_size, epoch, learning_rate, training_dataset):
        pass  # Train Network, TODO: Normalization Method 추가(Batch Norm 등)

    @abstractmethod
    def accuracy(self, test_dataset):
        pass  # Test Network, TODO: 자유도 높게

    @abstractmethod
    def __call__(self, input_data):
        pass  # Query To Network


class TFNetwork(Network):
    def __init__(self, layers, input_shape):
        self.layers = layers
        self.input = None
        self.output = self.init_layers(layers, input_shape)
        # TODO: Dropout, Summary 추가

    def init_layers(self, layers, input_shape):
        @dispatch(Dense, object)
        def init(node: Dense, in_flow):
            # 개더링 기반 입력 플로우 조정
            if isinstance(in_flow, list):
                if node.gathering == 'sum':
                    temp = in_flow
                    in_flow = in_flow[0]
                elif node.gathering == 'concat':
                    in_flow = tf.concat(1, in_flow)
                else:
                    exit()  # TODO: 예외처리, 개더링 명시 X

            # 파라미터 초기화
            node.weight = tf.Variable(tf.truncated_normal(shape=[
                in_flow.get_shape().as_list()[-1], node.units
            ]) / tf.sqrt(float(layer.units))/2 if node.activate == tf.nn.relu else 1)
            node.bias = tf.Variable(tf.truncated_normal(shape=[node.units]))

            # 레이어 출력값 반환
            def result(x):
                return node.activate(tf.matmul(x, layer.weight)+layer.bias)
            return result(in_flow) if node.gathering != 'sum' else sum([result(i) for i in temp])

        @dispatch(Conv2D, object)
        def init(node: Conv2D, in_flow):
            # 개더링 기반 입력 플로우 조정
            if isinstance(in_flow, list):
                if node.gathering == 'sum':
                    temp = in_flow
                    in_flow = in_flow[0]
                elif node.gathering == 'concat':
                    in_flow = tf.concat(2, in_flow)
                else:
                    exit()  # TODO: 예외처리, 개더링 명시 X

            # 파라미터 초기화
            node.weight = tf.Variable(tf.truncated_normal(shape=[
                *node.filter_size, in_flow.get_shape().as_list()[-1], node.filters
            ])/tf.sqrt(float(node.filters))/2 if node.activate == tf.nn.relu else 1)
            node.bias = tf.Variable(tf.random_normal(shape=[node.filters]), name='bias')

            # 레이어 출력값 반환
            def result(x):
                return node.activate(tf.nn.conv2d(x, node.weight, node.stride, node.padding))
            return result(in_flow) if node.gathering != 'sum' else sum([result(i) for i in temp])

        @dispatch(Pooling, object)
        def init(node: Pooling, in_flow):
            # 개더링 기반 입력 플로우 조정
            if isinstance(in_flow, list):
                if node.gathering == 'sum':
                    temp = in_flow
                    in_flow = in_flow[0]
                elif node.gathering == 'concat':
                    in_flow = tf.concat(2, in_flow)
                else:
                    exit()  # TODO: 예외처리, 개더링 명시 X

            # 파라미터 초기화

            # 레이어 출력값 반환
            def result(x):
                return node.activate(node.pooling(x, node.size, node.stride, node.padding))

            return result(in_flow) if node.gathering != 'sum' else sum([result(i) for i in temp])

        self.input = tf.placeholder(tf.float32, shape=[None, *input_shape])
        flow = self.input
        for layer in layers:
            if isinstance(layer, list):
                flow = [init(node, flow) for node in layer]
            else:
                flow = init(layer, flow)
        return flow

    def train(self, optimize, loss_fn, batch_size, epoch, learning_rate, training_dataset):
        # define placeholder
        object_output = tf.placeholder(tf.float32, [None, len(training_dataset[0][-1])])

        # define loss function
        with tf.name_scope('loss') as scope:
            if loss_fn == 'mse':
                loss = tf.reduce_mean(tf.square(object_output - self.output), name='MSE-Loss')
            elif loss_fn == 'cross-entropy':
                loss = -tf.reduce_sum(object_output * tf.log(tf.clip_by_value(self.output, 1e-30, 1.0)),
                                      name='cross-entropy-Loss')
            else:
                loss = None
                print('error : loss Function not defined')
                exit()

        # define train optimizer
        print(loss)
        if optimize == 'gradient-descent':
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        elif optimize == 'adam':
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        elif optimize == 'rms-prop':
            train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
        else:
            train_step = None
            print('optimizer not defined')
            exit()

        #
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            for _ in range(epoch):
                batch = []
                for __ in range(batch_size):
                    batch.append(choice(training_dataset))
                x_batch = [b[0] for b in batch]
                y_batch = [b[1] for b in batch]

                _ = sess.run(train_step, feed_dict={
                    self.input: x_batch,
                    object_output: y_batch
                    #  \, self.drop_p: drop_p
                })

    def accuracy(self, test_dataset):
        pass

    def __call__(self, input_data, model_path=None):
        pass

    def __str__(self):
        pass

