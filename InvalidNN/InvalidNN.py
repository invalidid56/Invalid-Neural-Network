'''
Invalid Neural Network v 0.02
tensorflow 에서 신경망을 빠르게 작성하고, 편하게 테스트하기 위한 모듈입니다.
Layer과 이를 상속받는 클래스들로 레이어를 정의하고, NeuralNetwork 객체를 만드십시오.
'''

import tensorflow as tf
import tensorflow.contrib as tf_c
from random import choice


def mul(*args):
    result = 1
    for i in args:
        result *= i
    return result


# Layers(추가사항: 정규화, 규제 레이어 추가, 레이어 생성시 생성메서드 추가 입력)
class Layer(object):  # Meta class for layers
    '''
    class Layer
    기본적인 레이어 클래스입니다.
    :param name: 레이어의 이름을 설정합니다(시각화나 name scoping)
    :param activate_fn: 레이어의 출력에 사용할 활성화 함수를 설정합니다.
    :param dropout: 레이어의 드롭아웃 적용 여부를 결정합니다.
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
    '''
    전결합층 레이어 클래스입니다.
    :param units: 레이어의 출력 유닛 개수를 설정합니다
    '''
    def __init__(self, name, activate_fn, units, dropout=False):
        super().__init__(name, activate_fn, dropout)  # init super
        self.units = units  # define layer's output units


class Conv2D(Layer):
    '''
    2D 입력에 대한 합성곱층 레이어 클래스입니다.
    :param filters: 레이어가 사용할 필터의 개수를 설정합니다(감지할 특성, 사용할 커널)
    :param filter_shape: 레이어가 사용할 필터의 모양을 설정합니다([height, weight])
    :param stride: 필터에 적용할 보폭(stride)를 설정합니다([h_stride, w_stride])
    :param padding: 필터에 적용할 패딩 방식을 설정합니다('valid', 'same')
    '''
    def __init__(self, name, activate_fn, filters, filter_shape, stride, padding, dropout=False):
        super().__init__(name, activate_fn, dropout)  # init super
        self.filters = filters  # define layer's output kernels
        self.filter_shape = filter_shape  # define filter's shape
        self.stride = stride  # define filter's stride
        self.padding = padding.upper()  # define layer's padding method


class Pooling(Layer):
    '''
    2D 입력에 대한 풀링층 레이어 클래스입니다
    :param pooling: 풀링 방식을 설정합니다('avg', 'max')
    :param stride: 풀링 윈도우의 보폭(stride)를 설정합니다([h_stride, w_stride])
    :param size: 풀링 윈도우의 크기를 설정합니다([height, weight])
    :param padding: 적용할 패딩 방식을 설정합니다('valid', 'same')
    '''
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
    뉴럴 네트워크 클래스입니다. Layer 타입의 인스턴스들을 입력받아 이를 연결합니다.
    :param layers: Layer 타입의 인스턴스들의 collection입니다
    :param input: 입력 레이어의 유닛의 개수입니다.
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
                    if l != 0 and (isinstance(layers[l-1], Conv2D) or isinstance(layers[l-1], Pooling)):
                        # 랭크 크기 비교로 수정
                        flow = tf.reshape(flow, [-1, mul(*flow.get_shape().as_list()[-3:])])
                    layer.weight = tf.Variable(
                        tf.random_normal([input[0] if l==0 else tf.to_int32(flow.shape[-1]), layer.units])/tf.sqrt(float(layer.units)/2),
                        name='weight',
                        trainable=True
                    )
                    layer.bias = tf.Variable(tf.random_normal(shape=[1]), name='bias')
                    flow = tf.matmul(flow, layer.weight)
            elif isinstance(layer, Conv2D):
                with tf.name_scope(layer.name) as scope:
                    layer.filter = tf.Variable(
                        tf.random_normal([*layer.filter_shape, tf.to_int32(flow.shape[-1]), layer.filters])/tf.sqrt(float(layer.filters)/2),
                        name = "filter",
                        trainable=True
                    )
                    layer.bias = tf.Variable(tf.random_normal(shape=[layer.filters]), name='bias')
                    flow = tf.nn.conv2d(flow, layer.filter, [1, *layer.stride, 1], layer.padding)
            elif isinstance(layer, Pooling):
                with tf.name_scope(layer.name) as scope:
                    if layer.pooling == 'max':
                        method = tf.nn.max_pool
                    elif layer.pooling == 'avg':
                        method = tf.nn.avg_pool
                    else:
                        print('error')
                    flow = method(flow, [1, *layer.size, 1], [1, *layer.stride, 1], layer.padding, name='pooling')
            else:
                print('layer not defined')
                exit()
            flow = activate_function[layer.activate](flow)
            if layer.dropout:
                flow = tf.nn.dropout(flow, self.drop_prob)
        self.output = flow
        self.saver = tf.train.Saver()

    def __getitem__(self, item):
        return self._layers[item]

    def __len__(self):
        return len(self._layers)

    def query(self, input, model_path='./'):
        '''
        function query(input, model_path): 질의 메서드입니다.
        :param input: 뉴럴 네트워크에 입력할 값입니다(n-d tensor혹은 그로 컨버팅 가능한 인스턴스).
        :param model_path: 모델 파일이 저장된 경로입니다. 기본값은 현재 경로입니다
        :return: 뉴럴 네트워크에서 input을 계산한 값이 반환됩니다
        '''
        with tf.Session() as sess:
            self.saver.restore(sess, model_path+'model.ckpt')
            return sess.run(self.output, feed_dict={self.input_data: [input], self.drop_prob: 1.}).reshape(-1).tolist()

    def train(self, train_data, batch_size, loss_function, optimizer, learning_rate, epoch, model_path = './', dropout_p=1.0):
        '''
        function train(...): 훈련 메서드입니다
        :param train_data: 훈련에 사용될 데이터세트입니다([x, y]의 collection)
        :param batch_size: 배치 하나의 사이즈입니다
        :param loss_function: 사용할 오차함수를 결정합니다('least-mean', 'cross-entropy')
        :param optimizer: 사용할 옵티마이저를 결정합니다('gradient-descent', 'adam', 'rms-prop')
        :param learning_rate: 학습률을 결정합니다
        :param epoch: 반복 횟수를 결정합니다
        :param model_path: 요약 파일과 모델 파일이 저장될 경로를 결정합니다
        :param dropout_p: 드롭아웃 확률을 결정합니다
        :return:
        '''
        # define placeholder
        object_output = tf.placeholder(tf.float32, [None, len(train_data[0][-1])])

        # define loss function
        with tf.name_scope('loss') as scope:
            if loss_function == 'least-square':
                loss = tf.reduce_mean(tf.square(object_output - self.output), name='least-square')
            elif loss_function == 'cross-entropy':
                loss = -tf.reduce_sum(object_output * tf.log(tf.clip_by_value(self.output, 1e-30, 1.0)), name='cross-entropy')
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

        with tf.Session() as sess:
            train_writter = tf.summary.FileWriter(model_path, sess.graph)
            sess.run(init)

            for _ in range(epoch):
                batch = []
                for __ in range(batch_size):
                    batch.append(choice(train_data) )
                x_batch = [b[0] for b in batch]
                y_batch = [b[1] for b in batch]

                summary, _ = sess.run([merged, train_step], feed_dict={
                        self.input_data: x_batch,
                        object_output: y_batch,
                        self.drop_prob: dropout_p
                    })
                train_writter.add_summary(summary)
            self.saver.save(sess, model_path+'model.ckpt')

            print('train progress finished')
