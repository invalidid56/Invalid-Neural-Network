from InvalidNN import invalidnn as inv
from InvalidNN.utill import pretreatment


layers = [
    inv.Conv2D(name='Conv_1', activate_fn='relu',  filters=64, filter_shape=[5, 5], stride=[1, 1], padding='same'),
    inv.Pooling(name='Pooling_1', pooling='max', stride=[2, 2], size=[2, 2], padding='same'),
    inv.Conv2D('Conv_2', 'relu', 64, [5, 5], [1, 1], 'same'),
    inv.Pooling('Pooling_2', 'max', [3, 3], [2, 2], 'same'),
    inv.Conv2D('Conv_3', 'relu', 128, [3, 3], [1, 1], 'same'),
    inv.Conv2D('Conv_3', 'relu', 128, [3, 3], [1, 1], 'same'),
    inv.Conv2D('Conv_3', 'relu', 128, [3, 3], [1, 1], 'same'),
    inv.Dense(name='Dense_1', activate_fn='sigmoid', units=384, dropout=True),
    inv.Dense('Dense_2: Softmax', 'softmax', 10)
]


train_data = [[x_train[i], y_train[i]] for i in range(len(x_train))]
test_data = [[x_test[i], y_test[i]] for i in range(len(x_test))]

train_data = pretreatment.one_hot(train_data, (0, 10), value=(0.01, 0.09))
test_data = pretreatment.one_hot(test_data, (0, 10), value=(0.01, 0.09))

cifar_net = inv.NeuralNetwork(layers, [32, 32, 3])

cifar_net.train(train_data, 128, 'cross-entropy', 'adam', 1e-4, dropout_p=0.7, epoch=10000)

print('no running time err')
