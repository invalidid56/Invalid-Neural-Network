from InvalidNN import invalidnn as inv
from InvalidNN.utill import pretreatment
import tensorflow as tf


train_dataset, test_dataset = pretreatment.mnist_download(dataset_dir='/home/invalidid/Datasets/MNIST')

sample_network = [
    inv.Dense('fc_1', tf.nn.sigmoid, 200),
    [inv.Dense('fc_2', tf.nn.softmax, 10), inv.Dense('fc_3', tf.nn.softmax, 10)],
    inv.Dense('fc_4', tf.nn.softmax, 10, gathering='sum')
]

mynet = inv.TFNetwork(sample_network, [784])

mynet.train('gradient-descent', 'mse', 10, 5, 0.05, train_dataset)
