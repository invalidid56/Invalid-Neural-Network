import tensorflow as tf
import tensorflow.contrib as tfc


def mul(*args):
    k = 1
    for a in args:
        k *= a
    return k


class Layer:  # 얘는 객체 속성 repr? 하면 layer(layer : 'FC', activate_function : 'sigmoid' ~~~) 이렇게 나오게
    def __init__(self, layer: str, activate_function: str, dropout: int = None,):
        # 범용 속성
        self.layer = layer
        self.activate = activate_function
        self.dropout = dropout


class FullyConnected(Layer):
    def __init__(self, activate_function, units, dropout=0):
        super().__init__('FullyConnected', activate_function, dropout)

        self.units = units


class Conv(Layer):
    def __init__(self, activate_function, padding:str, stride, filters, filter_shape, dropout=0):
        super().__init__('Conv', activate_function, dropout)

        self.filter_shape = filter_shape
        self.channels = filters

        self.padding = padding.upper()
        self.stride = stride


class Pooling(Layer):
    def __init__(self, pooling, stride, padding:str, size, activate_function=None, dropout=0):
        super().__init__('Pooling', activate_function, dropout)

        self.pooling = pooling
        self.stride = stride
        self.padding = padding.upper()

        self.size = size
        self.channels = 0


class NeuralNetwork:
    def __init__(self, layers, input_units):  # 신경망 검사, 가중치/필터 초기화, 그래프 작성
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
        }  # define activate function dict

        initializer = {
            'xavier': tfc.layers.xavier_initializer,
            'xavier_conv': tfc.layers.xavier_initializer_conv2d,
            'variance_scaling': tfc.layers.variance_scaling_initializer,
        }

        if isinstance(input_units, int): input_units = [input_units]
        flow = self.input_data = tf.placeholder(tf.float32, [None, *input_units])
        self.drop_prob = tf.placeholder(tf.float32)

        for l, layer in enumerate(layers):
            if isinstance(layer, FullyConnected):
                with tf.name_scope(layer.layer) as scope:
                    # 초기화 오퍼레이터, 정규화 오퍼레이터 정의
                    init = tfc.layers.xavier_initializer()  # 확장 시에 선택 파라미터에 다른 오퍼레이터 추가
                    if not isinstance(layers[l-1 if l > 1 else 0], FullyConnected):
                        flow = tf.layers.flatten(flow)
                    norm = None  # 확장 시에 추가
                    # 그래프 작성
                    flow = tfc.layers.fully_connected(
                        inputs = flow,
                        num_outputs = layer.units,
                        activation_fn = activate_function[layer.activate],
                        weights_initializer = init,
                    )
                    tfc.layers.summarize_tensor(flow, layer.layer+'/flow')
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
                flow = method(flow, layer.size, layer.stride, layer.padding)
            elif isinstance(layer, Recurrent):
                pass
            else:
                print('layer not defined')
                exit()
            if layer.dropout:
                flow = tf.nn.dropout(flow, self.drop_prob)
        self.output = flow

    def train(self, training_dataset, batch_size, loss_function, optimizer, learning_rate, dropout=1.0, epoch=1):
        merged = tf.summary.merge_all()

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
        elif optimizer == 'rms-prop':
            train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
        else:
            train_step = None
            print('optimizer not defined')
            exit()

        # 변수 초기화, 저장 옵티마이저
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        # 신경망 학습
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter('C:\Temp' + '/train',
                                                 sess.graph)
            sess.run(init)

            for _ in range(epoch):
                dataset_length = len(training_dataset)

                for b in range(round(dataset_length/batch_size)):
                    batch = training_dataset[b*batch_size: (b+1)*batch_size]

                    x_batch = [i[0] for i in batch]
                    y_batch = [i[1] for i in batch]

                    summary, _ = sess.run([merged, train_step], feed_dict={
                        self.input_data: x_batch,
                        object_output: y_batch,
                        self.drop_prob: dropout
                    })

                    train_writer.add_summary(summary, b)

            save_path = self.saver.save(sess, "C:\Temp\model.ckpt")

            print("Model Saved")

    def query(self, input_data):
        # 질의
        with tf.Session() as sess:
            self.saver.restore(sess, "C:\Temp\model.ckpt")
            result = sess.run(self.output, feed_dict={self.input_data: [input_data], self.drop_prob: 1.0})
        return result

    def __getitem__(self, item):
        pass


