import tensorflow as tf


def mul(*args):
    result = 1
    for i in args:
        result *= i
    return result


class Layer(object):
    """
    기본 레이어 클래스
    """
    def __init__(self, name, activate_fn, dropout=False):
        self.name = name
        activate = {
            'sigmoid': tf.nn.sigmoid,
            'relu': tf.nn.relu,
            'relu6': tf.nn.relu6,
            'elu': tf.nn.elu,
            'softmax': tf.nn.softmax,
            'log-softmax': tf.nn.log_softmax,
            'softplus': tf.nn.softplus,
            'softsign': tf.nn.softsign,
            'tanh': tf.nn.tanh,
            None: lambda x: x
        }
        self.activate_fn = activate[activate_fn]
        self.dropout = dropout

    def __str__(self):
        attributes = [[name, keys] for name, keys in self.__dict__.items()]
        attributes.pop(0)
        result = '{}' + '(' + ('{}: {}, '*(len(attributes)-1)) + '{}: {})'
        attributes = sum(attributes, [])
        return result.format(self.name, *attributes)


class Dense(Layer):
    """
    전결합층 클래스
    """
    def __init__(self, name, activate_fn, units, dropout=False):
        super().__init__(name, activate_fn, dropout)  # init super
        self.units = units


class Conv2D(Layer):
    """
    2D 입력에 대한 합성곱층 레이어 클래스입니다.
    :param filters: 레이어가 사용할 필터의 개수를 설정합니다(감지할 특성, 사용할 커널)
    :param filter_shape: 레이어가 사용할 필터의 모양을 설정합니다([height, weight])
    :param stride: 필터에 적용할 보폭(stride)를 설정합니다([h_stride, w_stride])
    :param padding: 필터에 적용할 패딩 방식을 설정합니다('valid', 'same')
    """
    def __init__(self, name, activate_fn, filters, filter_shape, stride, padding, dropout=False):
        super().__init__(name, activate_fn, dropout)  # init super
        self.filters = filters  # define layer's output kernels
        self.filter_shape = filter_shape  # define filter's shape
        if sum([isinstance(s, int) for s in stride]) != len(stride):
            print('stride error')
            exit()
        self.stride = stride  # define filter's stride
        self.padding = padding.upper()  # define layer's padding method


class Pooling(Layer):
    """
    2D 입력에 대한 풀링층 레이어 클래스입니다
    :param pooling: 풀링 방식을 설정합니다('avg', 'max')
    :param stride: 풀링 윈도우의 보폭(stride)를 설정합니다([h_stride, w_stride])
    :param size: 풀링 윈도우의 크기를 설정합니다([height, weight])
    :param padding: 적용할 패딩 방식을 설정합니다('valid', 'same')
    """
    def __init__(self, name, pooling, stride, size, padding, activate_fn=None, dropout=False):
        super().__init__(name, activate_fn, dropout)
        self.pooling = pooling.lower()  # define layer's pooling method
        self.stride = stride  # define stride
        self.size = size  # define pooling window's size
        self.padding = padding.upper()


class Shortcut(object):
    pass


class WrongConnection(Exception):
    pass


class NeuralNetwork(object):
    def __init__(self, layers, input_shape):
        self._layers = layers

        if isinstance(input_shape, int):
            input_shape = [input_shape]
        self.input = tf.placeholder(tf.float32, [*input_shape])

        def init_layers(nodes, in_flow):
            flow = in_flow
            for l, layer in enumerate(nodes):
                if isinstance(layer, list):
                    flow = sum([
                        init_layers(sub_graph, flow) for sub_graph in layer
                    ])  # 리스트 안에 리스트?
                    continue

                elif isinstance(layer, Dense):
                    with tf.name_scope(layer.name) as scope:
                        if len(flow.get_shape().as_list()) > 2:
                            flow = tf.reshape(in_flow, [-1, mul(*in_flow.get_shape().as_list()[-3:])])

                        layer._weight = tf.Variable(
                            tf.random_normal([
                                in_flow.get_shape().as_list()[-1], layer.units
                            ]) / tf.sqrt(float(layer.units) / (2 if layer.activate_fn == 'relu' else 1)),
                            name='weight',
                            trainable=True
                        )
                        layer._bias = tf.Variable(tf.random_normal(shape=[1]), name='bias')

                        flow = tf.matmul(flow, layer._weight) + layer._bias

                elif isinstance(layer, Conv2D):
                    with tf.name_scope(layer.name) as scope:
                        if l > 0 and isinstance(nodes[l-1], Dense):
                            print('Wrong Connection')
                            exit()

                        layer._filter = tf.Variable(
                            tf.random_normal([
                                *layer.filter_shape, in_flow.get_shape().as_list()[-1], layer.filters
                            ]) / tf.sqrt(float(layer.filters) / (2 if layer.activate_fn == 'relu' else 1)),
                            name="filter",
                            trainable=True
                        )
                        layer._bias = tf.Variable(tf.random_normal(shape=[layer.filters]), name='bias')

                        flow = tf.nn.conv2d(flow, layer._filter, [1, *layer.stride, 1], layer.padding) + layer._bias

                elif isinstance(layer, Pooling):
                    with tf.name_scope(layer.name) as scope:
                        if isinstance(layers[l-1], Dense):
                            print('Wrong Connection')
                            exit()
                        if layer.pooling == 'max':
                            method = tf.nn.max_pool
                        elif layer.pooling == 'avg':
                            method = tf.nn.avg_pool
                        else:
                            print('Pooling Method Not Defined')
                            exit()
                        flow = method(flow, [1, *layer.size, 1], [1, *layer.stride, 1], layer.padding,  name='pooling')

                elif isinstance(layer, Shortcut):
                    continue

                flow = layer.activate_fn(flow)

            return flow
