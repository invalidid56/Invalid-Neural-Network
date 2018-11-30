# Import
import tensorflow as tf
from multipledispatch import dispatch
from abc import *
from random import choice


def variable_summary(var: tf.Tensor):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.to_float(tf.reduce_mean(tf.square(var-mean))))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# Node Class


class Node(object):
    @dispatch(str, object)
    def __init__(self, name, gathering=None):
        self.name = name
        self.gathering = gathering

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
    def train(self, optimize, loss_fn, batch_size, epoch, learning_rate, training_dataset, testing_dataset):
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
            with tf.name_scope(node.name):
                with tf.name_scope('weight'):
                    node.weight = tf.Variable(dtype=tf.float32, initial_value=(tf.random_normal([
                        in_flow.get_shape().as_list()[-1], node.units
                    ])/tf.sqrt(float(node.units))/(2 if node.activate == tf.nn.relu else 1)))
                    variable_summary(node.weight)
                with tf.name_scope('bias'):
                    node.bias = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[node.units]))
                    variable_summary(node.bias)

            # 레이어 출력값 반환
            def activate(x):
                return node.activate(tf.matmul(x, node.weight)+node.bias)
            if node.gathering != 'sum':
                result = activate(in_flow)
            else:
                result = sum([activate(i) for i in temp])
            with tf.name_scope(node.name):
                with tf.name_scope('activate'):
                    variable_summary(result)
            return result

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

        self.input = tf.placeholder(tf.float32, shape=[None, *input_shape], name='input_placeholder')
        flow = self.input
        for layer in layers:
            if isinstance(layer, list):
                flow = [init(node, flow) for node in layer]
            else:
                flow = init(layer, flow)
        return flow

    def train(self, optimize, loss_fn, batch_size, epoch, learning_rate, training_dataset, testing_dataset,
              summary_path='.', model_path='.', print_progress=False):
        # 목표 출력값 플레이스홀더 정의
        object_output = tf.placeholder(tf.float32, [None, len(training_dataset[0][-1])])

        # 오차함수 정의 TODO: 더 많은 오차함수, 자유도 높이
        with tf.name_scope('loss') as scope:
            if loss_fn == 'mse':
                loss = tf.reduce_mean(tf.square(object_output - self.output))
                tf.summary.scalar('MSE-loss', loss)
            elif loss_fn == 'cross-entropy':
                loss = -tf.reduce_sum(object_output * tf.log(tf.clip_by_value(self.output, 1e-30, 1.0)),
                                      name='cross-entropy-Loss')
                tf.summary.scalar('CE-loss', loss)

            else:
                loss = None
                print('error : loss Function not defined')
                exit()

        # 요약 작성
        merged = tf.summary.merge_all()

        # 정확도 계산
        with tf.name_scope('accuracy'):
            correct = tf.equal(
                tf.argmax(self.output, 1), tf.argmax(object_output, 1)
            )
            accuracy = (tf.reduce_mean(tf.cast(correct, tf.float32)))*100
            tf.summary.scalar('accuracy', accuracy)

        # 옵티마이저 정의
        if optimize == 'gradient-descent':
            train_step = tf.train.GradientDescentOptimizer(learning_rate, name='GD-train').minimize(loss)
        elif optimize == 'adam':
            train_step = tf.train.AdamOptimizer(learning_rate, name='ADAM-train').minimize(loss)
        elif optimize == 'rms-prop':
            train_step = tf.train.RMSPropOptimizer(learning_rate, name='RMS-train').minimize(loss)
        else:
            train_step = None
            print('optimizer not defined')
            exit()

        #
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            train_writter = tf.summary.FileWriter

            sess.run(init)

            for step in range(epoch):
                batch = []
                for __ in range(batch_size):  # TODO: 제너레이터나 텐서플로우 내장 배치 추출
                    batch.append(choice(training_dataset))
                x_batch = [b[0] for b in batch]
                y_batch = [b[1] for b in batch]

                _, summary = sess.run([train_step, merged], feed_dict={
                    self.input: x_batch,
                    object_output: y_batch
                    #  \, self.drop_p: drop_p
                })

                if (step % 10) == 0:
                    summary, acc = sess.run([merged, accuracy],
                                            feed_dict={
                                                self.input: 0,
                                                object_output: 0
                                            })


                if (step%1000) == 0:
                    pass

    def accuracy(self, test_dataset):
        pass

    def __call__(self, input_data, model_path=None):
        pass

    def __str__(self):
        pass

