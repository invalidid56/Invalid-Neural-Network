from InvalidNN import invalidnn as inv
from InvalidNN.utill import pretreatment
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data

layers = [
    inv.Conv('relu', 'same', [1, 1], 64, [5, 5]),
    inv.Pooling('max', [1, 2, 2, 1], 'same', [1, 3, 3, 1]),
    inv.Conv('relu', 'same', [1, 1], 64, [5, 5]),
    inv.Pooling('max', [1, 2, 2, 1], 'same', [1, 3, 3, 1]),
    inv.Conv('relu', 'same', [1, 1], 128, [3, 3]),
    inv.Conv('relu', 'same', [1, 1], 128, [3, 3]),
    inv.Conv('relu', 'same', [1, 1], 128, [3, 3]),
    inv.FullyConnected('relu', 384),
    inv.FullyConnected('softmax', 10, dropout=True)
]

(x_train, y_train), (x_test, y_test) = load_data()
train_data = [[x_train[i], y_train[i]] for i in range(len(x_train))]
test_data = [[x_test[i], y_test[i]] for i in range(len(x_test))]

train_data = pretreatment.one_hot(train_data, (0, 10), value=(0.01, 0.09))
test_data = pretreatment.one_hot(test_data, (0, 10), value=(0.01, 0.09))

cifar_net = inv.NeuralNetwork(layers, [32, 32, 3])

cifar_net.train(train_data, 128, 'cross-entropy', 'adam', 1e-4, dropout=0.7, epoch=10000)

print('no running time err')
