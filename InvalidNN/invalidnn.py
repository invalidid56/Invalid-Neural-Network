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
        self.output = self.init_layers(layers, input_shape)
        # TODO: Dropout, Summary 추가

    def init_layers(self, layers, input_shape):
        @dispatch(Dense, object)
        def init(layer, in_flow):
            # 개더링 기반 입력 플로우 조정
            if isinstance(in_flow, list):
                if layer.gathering == 'sum':
                    temp = in_flow
                    in_flow = in_flow[0]
                elif layer.gathering == 'concat':
                    in_flow = tf.concat(1, in_flow)
                else:
                    exit()  # TODO: 예외처리, 개더링 명시 X

            # 파라미터 초기화
            layer.weight = tf.truncated_normal(shape=[
                in_flow.get_shape().as_list()[-1], layer.units
            ]) / tf.sqrt(float(layer.units))
            layer.bias = tf.truncated_normal(shape=[layer.units])

            # 레이어 출력값 반환
            def result(x):
                return layer.activate(tf.matmul(x, layer.weight)+layer.bias)
            return result(in_flow) if layer.gathering != 'sum' else sum([result(i) for i in temp])

        flow = tf.placeholder(tf.float32, shape=[None, *input_shape])
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
            if loss_fn == 'least-square':
                loss = tf.reduce_mean(tf.square(object_output - self.output), name='least-square')
            elif loss_fn == 'cross-entropy':
                loss = -tf.reduce_sum(object_output * tf.log(tf.clip_by_value(self.output, 1e-30, 1.0)),
                                      name='cross-entropy')
            else:
                loss = None
                print('error : loss Function not defined')
                exit()

        # define train optimizer
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

    def __call__(self, input_data):
        pass

