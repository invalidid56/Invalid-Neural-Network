import invalidNN as inv
import tensorflow as tf
import csv

training_data= []

f = open("mnist_train.csv")
for line in csv.reader(f):
    training_data.append(line)

def one_hot(max, data):
    return [0.01 if i != data else 0.99 for i in range(max)]

def preprocess(dataset):
    result = []
    for sample in dataset:
        result.append([[int(s)/256 for s in sample[1:]], one_hot(10, int(sample[0]))])
    return result

training_data = preprocess(training_data)

mnist_network = [
    inv.FullyConnected('sigmoid', 200),
    inv.FullyConnected('softmax', 10)
]
AlphaGO = inv.NeuralNetwork(mnist_network, input_units=784)

AlphaGO.train(training_data, 100, 'least-square', 'gradient-descent', 0.01, epoch=100)
