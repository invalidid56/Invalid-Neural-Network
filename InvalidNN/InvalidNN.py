'''
module docs
'''

import tensorflow as tf
import tensorflow.contrib as tf_c
from random import choice

# Layers(추가사항: 정규화, 규제 레이어 추가, 레이어 생성시 생성메서드 추가 입력)
class Layer(object):  # Meta class for layers
    '''
    class Layer
    '''
    def __init__(self, name, activate_fn, dropout=False):
        self.name = name  # Layer's name: for name scoping
        self.activate = activate_fn  # activate function
        self.dropout = dropout  # apply dropout or not

    def __str__(self):
        attributes = [[name, keys] for name, keys in self.__dict__.items()]
        attributes.pop(0)
        result = '{}' + '(' + ('{}: {}, '*(len(attributes)-1)) + '{}: {})'
        attributes = sum(attributes, [])
        return result.format(self.name, *attributes)


class Dense(Layer):
    def __init__(self, name, activate_fn, units, dropout=False):
        super().__init__(name, activate_fn, dropout)  # init super
        self.units = units  # define layer's output units


class Conv2D(Layer):
    def __init__(self, name, activate_fn, filters, filter_shape, stride, padding, dropout=False):
        super().__init__(name, activate_fn, dropout)  # init super
        self.filters = filters  # define layer's output kernels
        self.filter_shape = filter_shape  # define filter's shape
        self.stride = stride  # define filter's stride
        self.padding = padding.upper()  # define layer's padding method


class Pooling(Layer):
    def __init__(self, name, pooling, stride, size, padding, activate_fn=None, dropout=False):
        super().__init__(name, activate_fn, dropout)
        self.pooling = pooling.lower()  # define layer's pooling method
        self.stride = stride  # define stride
        self.size = size  # define pooling window's size
        self.padding = padding.upper()


# Neural Networks
class NeuralNetwork:
    '''
    class NeuralNetwork
    '''
    def __init__(self, layers, input):
        self._layers = layers
        # define methods
        activate_function = {
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

        # define placeholders
        if isinstance(input, int): input = [input]
        flow = self.input_data = tf.placeholder(tf.float32, [None, *input])
        batch_size = flow.shape[0]
        self.drop_prob = tf.placeholder(tf.float32)

        # initialize layers
        for l, layer in enumerate(layers):
            if isinstance(layer, Dense):
                with tf.name_scope(layer.name) as scope:
                    if l>0 and (isinstance(layers[l-1], Conv2D) or isinstance(layers[l-1], Pooling)):
                        # 랭크 크기 비교로 수정
                        flow = tf_c.layers.flatten(flow)
                    layer.weight = tf.Variable(
                        tf.random_normal([input[0] if l==0 else layers[l-1].units, layer.units]),
                        name='weight'
                    )
                    layer.bias = tf.Variable(tf.constant(-1.), name='bias')
                    flow = tf.matmul(flow, layer.weight) + layer.bias

                    tf.summary.tensor_summary(layer.name+'/weight', layer.weight)
                    tf.summary.tensor_summary(layer.name+'/bias', layer.bias)
                    tf.summary.tensor_summary(layer.name+'/flow', flow)
            elif isinstance(layer, Conv2D):
                with tf.name_scope(layer.name) as scope:
                    layer.filter = tf.Variable(name="filter",
                                                   shape=[*layer.filter_shape, flow.shape[-1], layer.filters],
                                                   initializer=tf_c.layers.xavier_initializer_conv2d())
                    layer.bias = tf.Variable(-1., [layer.filters], name='bias')
                    flow = tf.nn.conv2d(flow, layer.filter, [1, *layer.stride, 1], layer.padding) + layer.bias

                    tf.summary.tensor_summary(layer.name + '/filter', layer.filter)
                    tf.summary.tensor_summary(layer.name + '/bias', layer.bias)
                    tf.summary.tensor_summary(layer.name + '/flow', flow)
                    tf.summary.image(layer.name + '/flow_image', flow)
            elif isinstance(layer, Pooling):
                with tf.name_scope(layer.name) as scope:
                    if layer.pooling == 'max':
                        method = tf.nn.max_pool
                    elif layer.pooling == 'avg':
                        method = tf.nn.avg_pool
                    else:
                        print('error')
                    flow = method(flow, layer.size, layer.stride, layer.padding, name='padding')

                    tf.summary.tensor_summary(layer.name + '/flow', flow)
                    tf.summary.image(layer.name + '/flow_image', flow)
            else:
                print('layer not defined')
                exit()
            flow = activate_function[layer.activate](flow)
            if layer.dropout:
                flow = tf.nn.dropout(flow, self.drop_prob)
        self.output = flow


    def __getitem__(self, item):
        return self._layers[item]

    def __len__(self):
        return len(self._layers)

    def __str__(self):
        pass

    def query(self, input, model_path='.'):
        pass

    def train(self, train_data, batch_size, loss_function, optimizer, learning_rate, epoch, model_path = './', dropout_p=1.0):
        # define placeholder
        object_output = tf.placeholder(tf.float32, [None, len(train_data[0][-1])])

        # define loss function
        with tf.name_scope('loss') as scope:
            if loss_function == 'least-square':
                loss = tf.reduce_mean(tf.square(object_output - self.output), name='least-square')
            elif loss_function == 'cross-entropy':
                loss = -tf.reduce_sum(object_output * tf.log(self.output), name='cross-entropy')
            else:
                loss = None
                print('error : loss Function not defined')
                exit()

            tf_c.layers.summarize_tensor(loss, '/loss')

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
        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            train_writter = tf.summary.FileWriter(model_path, sess.graph)

            sess.run(init)

            for _ in range(epoch):
                batch = []
                for __ in range(batch_size):
                    batch.append(choice(train_data))
                x_batch = [b[0] for b in batch]
                y_batch = [b[1] for b in batch]

                summary, _ = sess.run([merged, train_step], feed_dict={
                        self.input_data: x_batch,
                        object_output: y_batch,
                        self.drop_prob: dropout_p
                    })

                train_writter.add_summary(summary)
            self.saver.save(sess, model_path+'\model.ckpt')

            print('train progress finished')
