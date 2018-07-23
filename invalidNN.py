import tensorflow as tf


def mul(*args):
    k = 1
    for a in args:
        k *= a
    return k


class layer:  # 얘는 객체 속성 repr? 하면 layer(layer : 'FC', activate_function : 'sigmoid' ~~~) 이렇게 나오게, 컨볼이랑 엪씨 분리 고려
    def __init__(self, layer: str, activate_function: str, units: int = None, dropout: int = None, padding:str = None, filter_shape: list = None, filters :int = None, stride:list = None, pooling = None):
        # 범용 속성
        self.layer = layer
        self.activate = activate_function
        self.units = units
        self.dropout = dropout

        # 레이어별 속성
        if layer == 'FC':
            self.weight = self.bias = None

        elif layer == 'Conv':
            self.filter = self.bias = None
            self.filter_shape = filter_shape
            self.filters = filters
            self.stride = stride
            self.padding = padding

        elif layer == 'Pooling':
            self.ksize = [1] + filter_shape + [1]
            self.stride = stride
            self.padding = padding
            self.pooling = pooling


class NeuralNetwork:
    def __init__(self, layers, input_units):
        # 초기화, 모델 생성도

        activate_functions = {
            'ReLU' : tf.nn.relu,
            'Sigmoid' : tf.nn.sigmoid,
            'Softmax' : tf.nn.softmax,
            None : lambda x: x
        }

        x = tf.placeholder(tf.float32, shape=[None, mul(*input_units)])

        batch_size = x.shape[0]

        for layer, l in enumerate(layers):
            if layer.layer == 'FC':  # 앞에 놈이 컨볼이면 리쉐잎잉
                layer.weight = tf.Variable(
                    tf.random_normal(shape=[layer.units, input_units if l == 0 else layers[l-1].units],
                                     mean=tf.sqrt(layer.units))
                )
                layer.bias = tf.Variable([-1])  #

                if l != 0 and (layer[l-1] == 'Conv' or layer[l-1] == 'Pooling'):
                    network = tf.reshape(network, [-1, mul(*network.shape)/batch_size])

                network = tf.matmul(layer.weight, x if l == 0 else network) + layer.bias

                network = activate_functions[layer.activate](network)

            if layer.layer is 'Conv':
                layer.filter = tf.Variable(
                    tf.random_normal(shape=[*layer.filter_shape, input_units[-1], layer.filters],
                                     mean=0.1)
                )
                layer.bias = tf.Variable([-1, *input_units])

                if l == 0:
                    network = tf.reshape(x, [-1, *input_units])

                network = tf.nn.conv2d(input=network, filter=layer.filter, strides=layer.stride, padding=layer.padding)

                network = activate_functions[layer.activate](network)

            if layer.layer is 'Pooling':
                pooling = {
                    'avg' : tf.nn.avg_pool,
                    'max' : tf.nn.max_pool
                }

                network = pooling[layer.pooling](network, layer.ksize, layer.strides, layer.padding)

                network = activate_functions[layer.activate](network)
