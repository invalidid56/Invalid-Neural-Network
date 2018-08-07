import tensorflow as tf
import tensorflow.contrib as tfc


def mul(*args):
    k = 1
    for a in args:
        k *= a
    return k


class Layer:  # 얘는 객체 속성 repr? 하면 layer(layer : 'FC', activate_function : 'sigmoid' ~~~) 이렇게 나오게
    def __init__(self, layer: str, activate_function: str, dropout: int = None):
        # 범용 속성
        self.layer = layer
        self.activate = activate_function
        self.dropout = dropout


class FullyConnected(Layer):
    def __init__(self, activate_function, units, dropout=0):
        super().__init__('FullyConnected', activate_function, dropout)

        self.units = units


class Conv(Layer):
    def __init__(self, activate_function, padding, stride, filters, filter_shape, dropout = 0):
        super().__init__('Conv', activate_function, dropout)

        self.filter_shape = filter_shape
        self.channels = filters

        self.padding = padding.capitalize()
        self.stride = stride


class Pooling(Layer):
    def __init__(self, pooling, stride, padding, size, activate_function=None, dropout=0):
        super().__init__('Pooling', activate_function, dropout)

        self.pooling = pooling
        self.stride = stride
        self.padding = padding

        self.size = size
        self.channels = 0


class NeuralNetwork:
    def __init__(self, layers, input_units):  # 신경망 검사, 가중치/필터 초기화, 그래프 작성
        # 변수 저장 옵티마이저
        self.saver = None

        activate_function = {
            'sigmoid': tf.nn.sigmoid,
            'relu': tf.nn.relu,
            'softmax': tf.nn.softmax,
            'tanh': tf.nn.tanh,
            None: lambda x: x
        }
        if isinstance(input_units, int):
            input_units = [input_units]
        self.input_data = tf.placeholder(tf.float32, [None, *input_units])
        flow = self.input_data
        self.layers = layers

        for l, layer in enumerate(layers):
            if isinstance(layer, FullyConnected):
                # 초기화 오퍼레이터, 정규화 오퍼레이터 정의
                init = tfc.layers.xavier_initializer()  # 확장 시에 선택 파라미터에 다른 오퍼레이터 추가
                norm = None  # 확장 시에 추가

                # 그래프 작성
                flow = tfc.layers.fully_connected(
                    inputs=flow,
                    num_outputs=layer.units,
                    activation_fn=activate_function[layer.activate],
                    weights_initializer=init
                )
            elif isinstance(layer, Conv):
                # 초기화 오퍼레이터, 정규화 오퍼레이터 정의
                init = tfc.layers.xavier_initializer()
                norm = None

                # 그래프 작성, 확장 시에 여러 컨볼루션 모델 적용
                flow = tfc.layers.convolution2d(
                    inputs=flow,
                    num_outputs=layer.channels,
                    kernel_size=layer.filter_shape,
                    activation_fn=activate_function[layer.activate],
                    weights_initializer=init,
                    padding=layer.padding,
                    stride=layer.stride
                )
            elif isinstance(layer, Pooling):
                method = tf.nn.max_pool if layer.pooling == 'max' else tf.nn.avg_pool
                flow = method(flow, layer.size, layer.stride)
            else:
                print('layer not defined')
                exit()
        self.output = flow

    def train(self, training_dataset, batch_size, loss_function, optimizer, learning_rate, epoch = 1):
        object_output = tf.placeholder(tf.float32, [None, len(training_dataset[0][1])])
        # 오차함수 정의
        if loss_function == 'least-square':
            loss = tf.reduce_mean(tf.square(object_output - self.output))
        elif loss_function == 'cross-entropy':
            loss = -tf.reduce_sum(object_output*tf.log(self.output))
        else:
            loss = None
            print('error : loss Function not defined')
            exit()

        # 옵티마이저 정의
        if optimizer == 'gradient-descent':
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        elif optimizer == 'adam':
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        else:
            train_step = None
            print('optimizer not defined')
            exit()

        # 변수 초기화, 저장 옵티마이저
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        # 신경망 학습
        with tf.Session() as sess:
            sess.run(init)

            for _ in range(epoch):
                dataset_length = len(training_dataset)

                for b in range(round(dataset_length/batch_size)):
                    batch = training_dataset[b*batch_size: (b+1)*batch_size]

                    x_batch = [i[0] for i in batch]
                    y_batch = [i[1] for i in batch]

                    sess.run(train_step, feed_dict={self.input_data: x_batch, object_output: y_batch})

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


