import tensorflow as tf
import tensorflow.contrib as tf_c
from random import choice


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
        self.drop_p = tf.placeholder(tf.float32)

        def init_layers(nodes, in_flow):
            flow = in_flow
            for l, layer in enumerate(nodes):
                if isinstance(layer, list):
                    flow = sum([
                        init_layers(sub_graph, flow) for sub_graph in (layer if isinstance(layer[0], list) else [layer])
                    ])
                    continue

                elif isinstance(layer, Dense):
                    with tf.name_scope(layer.name) as scope:
                        if len(flow.get_shape().as_list()) > 2:
                            flow = tf.reshape(flow, [-1, mul(*flow.get_shape().as_list()[-3:])])

                        layer._weight = tf.Variable(
                            tf.random_normal([
                                flow.get_shape().as_list()[-1], layer.units
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
                                *layer.filter_shape, flow.get_shape().as_list()[-1], layer.filters
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
                            method = None
                            print('Pooling Method Not Defined')
                            exit()
                        flow = method(flow, [1, *layer.size, 1], [1, *layer.stride, 1], layer.padding,  name='pooling')

                elif isinstance(layer, Shortcut):
                    continue

                flow = layer.activate_fn(flow)

                if layer.dropout:
                    flow = tf.nn.dropout(flow, self.drop_p)

            return flow

        self.output = init_layers(layers, self.input)

        self.saver = tf.train.Saver()

    def train(self, training_dataset, loss_fn, optimizer, learning_rate, batch_size, epoch, drop_p=1.0, model_path='.', summary_path='.'):
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
        tf.summary.scalar('loss', loss)

        # define train optimizer
        if optimizer == 'gradient-descent':
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        elif optimizer == 'adam':
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        elif optimizer == 'rms-prop':
            train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
        else:
            train_step = None
            print('optimizer not defined')
            exit()

        #
        init = tf.global_variables_initializer()
        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(summary_path + '', sess.graph)
            sess.run(init)

            for _ in range(epoch):
                batch = []
                for __ in range(batch_size):
                    batch.append(choice(training_dataset))
                x_batch = [b[0] for b in batch]
                y_batch = [b[1] for b in batch]

                summary, _ = sess.run([merged, train_step], feed_dict={
                    self.input: x_batch,
                    object_output: y_batch,
                    self.drop_p: drop_p
                })
                train_writer.add_summary(summary)
            self.saver.save(sess, model_path + '/' + 'model.ckpt')

    def query(self, input_data, model_path=None):
        with tf.Session() as sess:
            if model_path:
                self.saver.restore(sess, model_path+'/model.ckpt')
            else:
                sess.run(tf.global_variables_initializer())
            feed = {
                self.input: [input_data],
                self.drop_p: 1.
            }
            return sess.run(self.output, feed).tolist()
