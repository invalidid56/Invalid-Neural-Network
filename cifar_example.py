from InvalidNN import invalidnn as inv
from InvalidNN.utill import pretreatment
from InvalidNN.utill import test


layers = [
    inv.Conv2D(name='Conv_1', activate_fn='relu',  filters=64, filter_shape=[5, 5], stride=[1, 1], padding='same'),
    inv.Pooling(name='Pooling_1', pooling='max', stride=[2, 2], size=[2, 2], padding='same'),
    inv.Conv2D('Conv_2', 'relu', 64, [5, 5], [1, 1], 'same'),
    inv.Pooling('Pooling_2', 'max', [3, 3], [2, 2], 'same'),
    inv.Conv2D('Conv_3', 'relu', 128, [3, 3], [1, 1], 'same'),
    inv.Conv2D('Conv_3', 'relu', 128, [3, 3], [1, 1], 'same'),
    inv.Conv2D('Conv_3', 'relu', 128, [3, 3], [1, 1], 'same'),
    inv.Dense(name='Dense', activate_fn='sigmoid', units=384, dropout=True),
    inv.Dense('Softmax', 'softmax', 10)
]

train_data, test_data = pretreatment.cifar10_download('D:\Programming\Dataset\CIFAR-10')

cifar_net = inv.NeuralNetwork(layers, [32, 32, 3])

cifar_net.train(train_data, 128, 'cross-entropy', 'adam', 1e-4, dropout_p=0.7, epoch=10000)

acc = test.test_model(cifar_net, test_data)
print('정확도: ' + acc)
