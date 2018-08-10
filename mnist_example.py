from tensorflow.examples.tutorials.mnist import input_data
from InvalidNN import InvalidNN as inv
from InvalidNN.utill import pretreatment
from InvalidNN.utill import test

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
training_x, training_y = mnist.train.next_batch(60000)
training_x = pretreatment.data_normalization(training_x, 'min-max', (0.01, 0.99))
training_y = pretreatment.data_normalization(training_y, 'min-max', (0.01, 0.99))
training_data = [[training_x[i], training_y[i]] for i in range(60000)]

sample_network = [
    inv.FullyConnected('sigmoid', 200, dropout=True),
    inv.FullyConnected('softmax', 10)
]

mynet = inv.NeuralNetwork(sample_network, input_units=784)

mynet.train(training_data, 10, 'least-square', 'gradient-descent', 0.05, epoch=150)

print('정확도 : ', test.test(mynet, training_data[0:500]))


