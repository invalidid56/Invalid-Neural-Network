import tensorflow as tf
import tensorflow.contrib as tfc
from random import shuffle


def mul(*args):
    k = 1
    for a in args:
        k *= a
    return k


class layer:  # 얘는 객체 속성 repr? 하면 layer(layer : 'FC', activate_function : 'sigmoid' ~~~) 이렇게 나오게
    def __init__(self, layer: str, activate_function: str, dropout: int = None):
        # 범용 속성
        self.layer = layer
        self.activate = activate_function
        self.dropout = dropout


class FullyConnected(layer):
    def __init__(self, activate_function, units, dropout = 0):
        super().__init__('FullyConnected', activate_function, dropout)

        self.units = units


class Conv(layer):
    def __init__(self, activate_function, padding, stride, filters, filter_shape, dropout = 0):
        super().__init__('Conv', activate_function, dropout)

        self.filter_shape = filter_shape
        self.channels = filters

        self.padding = padding
        self.stride = stride


class Pooling(layer):
    def __init__(self, pooling, stride, padding, size, activate_function = None, dropout = 0):
        super().__init__('Pooling', activate_function, dropout)

        self.pooling = pooling
        self.stride = stride
        self.padding = padding

        self.size = size
        self.channels = 0


class NeuralNetwork:
    def __init__(self, layers, input_units): # 신경망 검사, 가중치/필터 초기화, 그래프 작성
        activate_function = {
            'sigmoid' : tf.nn.sigmoid,
            'relu' : tf.nn.relu,
            'softmax' : tf.nn.softmax,
            'tanh' : tf.nn.tanh,
            None : lambda x: x
        }
        if isinstance(input_units, int):
            input_units = [input_units]
        self.input_data = tf.placeholder(tf.float32, [None, *input_units])
        batch_size = self.input_data.shape[0]
        flow = self.input_data
        self.layers = layers

        for l, layer in enumerate(layers):
            if isinstance(layer, FullyConnected):
                # 초기화 오퍼레이터, 정규화 오퍼레이터 정의
                init = tfc.layers.xavier_initializer()  # 확장 시에 선택 파라미터에 다른 오퍼레이터 추가
                norm = None  # 확장 시에 추가

                # 그래프 작성
                if isinstance(layers[l-1], Conv) or isinstance(layers[l-1], Pooling):
                    flow = tf.reshape(flow, [-1, mul(layers[l-1].shape)/batch_size])

                flow = tfc.layers.fully_connected(
                    inputs = flow,
                    num_outputs = layer.units,
                    activation_fn = activate_function[layer.activate],
                    weights_initializer= init
                )

            elif isinstance(layer, Conv):
                # 필터 초기화
                layer.filter = tf.Variable(
                    tf.random_normal([*layer.filter_shape, input_units[-1] if l ==0 else layers[l-1].channels, layer.channels])
                )
                layer.bias = tf.Variable(tf.random_normal([layer.channels]))

                # 그래프 작성
                flow = activate_function[layer.activate](
                    tf.nn.conv2d(network, layer.filter, layer.stride, layer.padding) + layer.bias
                )

            elif isinstance(layer, Pooling):
                layer.channels = layers[l-1].channels

                pooling = tf.nn.avg_pool if layer.pooling == 'avg' else tf.nn.max_pool
                flow = activate_function[layer.activate](pooling(flow, layer.size, layer.stride, layer.padding))

        self.output = flow

    def train(self, training_dataset, batch_size, loss_function, optimizer, learning_rate, epoch = 1):
        object_output = tf.placeholder(tf.float32, [None, len(training_dataset[0][1])])
        # 오차함수 정의
        if loss_function == 'least-square':
            loss = tf.reduce_mean(tf.square(object_output - self.output))
        elif loss_function == 'cross-entopy':
            loss = -tf.reduce_sum(object_output*tf.log(self.output))

        # 옵티마이저 정의
        if optimizer == 'gradient-descent':
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        elif optimizer == 'adam':
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # 변수 초기화 옵티마이저
        init = tf.global_variables_initializer()

        # 변수 저장 옵티마이저
        self.saver = tf.train.Saver()

        # 신경망 학습
        with tf.Session() as sess:
            sess.run(init)

            for _ in range(epoch):
                # 데이터세트 섞기
                dataset_length = len(training_dataset)

                # 배치 사이즈만큼 나눠서 학습
                for b in range(round(dataset_length/batch_size)):
                    batch = training_dataset[b*batch_size : (b+1)*batch_size]

                    x_batch = [i[0] for i in batch]
                    y_batch = [i[1] for i in batch]

                    sess.run(train_step,
                            feed_dict={self.input_data: x_batch, object_output: y_batch})

            print(sess.run(self.output, feed_dict={self.input_data: [training_dataset[0][0]]}))

            save_path = self.saver.save(sess, "C:\Temp\model.ckpt")
            print("Model Saved")


    def query(self, input_data):
        # 질의
        with tf.Session() as sess:
            self.saver.restore(sess, "C:\Temp\model.ckpt")
            result = sess.run(self.output, feed_dict={self.input_data: [input_data]})
        return result

    def __getitem__(self, item):
        pass


