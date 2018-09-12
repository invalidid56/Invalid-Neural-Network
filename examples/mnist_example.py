from InvalidNN import invalidnn_new as inv
from InvalidNN.utill import pretreatment
from InvalidNN.utill import test
import gzip

train_dataset, test_dataset = pretreatment.mnist_download()

sample_network = [
    inv.Dense('fc_1', 'sigmoid', 200, dropout=True),
    inv.Dense('fc_2', 'softmax', 10)
]

mynet = inv.NeuralNetwork(sample_network, [None, 784])

mynet.train(
    train_dataset,
    batch_size=100,
    loss_fn='least-square',
    optimizer='gradient-descent',
    learning_rate=0.05,
    epoch=5000,
    model_path='.',
    summary_path='./summary',
    drop_p=0.7)
