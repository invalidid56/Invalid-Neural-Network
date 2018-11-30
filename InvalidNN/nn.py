from InvalidNN.core import *
import tensorflow as tf  # TODO: TORCH
import multipledispatch


ACTIVATE = {
    'relu': tf.nn.relu
}


class Layer(Node):
    def __init__(self, name, scope, activate, dropout=False):
        super().__init__(name, scope)
        self.activate = ACTIVATE[activate]
        self.dropout = dropout


class Dense(Layer):
    def __init__(self, name, scope, activate, units, dropout=False):
        super().__init__(name, scope, activate, dropout)
        self.units = units
        self.weight = None
        self.bias = None

    def func(self, k):
        act = self.activate(
            tf.matmul(self.weight, k) + self.bias
        )
        return act if not self.dropout else tf.nn.dropout(act, keep_prob=self.scope.drop_p)


class NeuralNetwork(Graph):
    @abstractmethod
    def init_layers(self, layers, input_shape):
        pass  # Initialize Layers

    @abstractmethod
    def train(self, optimize, loss_fn, batch_size, epoch, learning_rate,
              train_data_generator, test_data_generator):
        pass  # Train Network, TODO: Normalization Method 추가(Batch Norm 등)

    @abstractmethod
    def accuracy(self, test_data_generator):
        pass  # Test Network, TODO: 자유도 높게

    @abstractmethod
    def __call__(self, input_data):
        pass  # Query To Network


class TFNeuralNetwork(Graph):
    def __init__(self, name, scope, nodes, input_shape):
        super().__init__(nodes, name, scope)
        self.input_shape = input_shape
        self.drop_p = tf.placeholder(tf.float32)
        self.input_generator = None # TODO: TF에서 입력으로 제너레이터를 받는 방법에 리서치 필요.

    def init_layers(self, layers, in_flow):
        # 초기화 메서드 정
        @dispatch(Dense, object)
        def init(node, x):
            # 파라미터 초기화
            with tf.name_scope(node.name):
                with tf.name_scope('weight'):
                    node.weight = tf.Variable(dtype=tf.float32, initial_value=(tf.random_normal([
                        in_flow.get_shape().as_list()[-1], node.units
                    ]) / tf.sqrt(float(node.units)) / (2 if node.activate == tf.nn.relu else 1)))
                    variable_summary(node.weight)
                with tf.name_scope('bias'):
                    node.bias = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[node.units]))
                    variable_summary(node.bias)
            # 그래프 연결
            return node(x)

        #
        for layer in layers:
            flow = 0


