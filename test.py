import invalidNN as inv
import gzip

with gzip.open("mnist_train.gz", 'r') as f:
    training_data = [x.decode('utf8').strip() for x in f.readlines()]
    f.close()

with gzip.open("mnist_test.gz", 'r') as f:
    test_data = [x.decode('utf8').strip() for x in f.readlines()]
    f.close()

def one_hot(max, data):
    return [0.01 if i != data else 0.99 for i in range(max)]

def preprocess(dataset):
    dataset = [[
        int(list(sample).pop(0)), [int(atoms) for atoms in sample.split(',')[1:]]
    ] for sample in dataset]

    dataset = [[
        [atoms/256 for atoms in sample[1]], one_hot(10, sample[0])
    ] for sample in dataset]
    return dataset

training_data = preprocess(training_data)
preprocess(test_data)

mnist_network = [
    inv.FullyConnected('sigmoid', 200),
    inv.FullyConnected('softmax', 10)
]

AlphaGO = inv.NeuralNetwork(mnist_network, input_units=784)


AlphaGO.train(training_data, 10, 'least-square', 'gradient-descent', 0.01)
