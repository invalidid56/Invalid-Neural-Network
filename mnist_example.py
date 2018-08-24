from InvalidNN import invalidnn as inv
from InvalidNN.utill import pretreatment
from InvalidNN.utill import test
import gzip

train_dataset, test_dataset = pretreatment.mnist_download()

sample_network = [
    inv.Dense('fc_1', 'sigmoid', 200, dropout=True),
    inv.Dense('fc_2', 'softmax', 10)
]

mynet = inv.NeuralNetwork(sample_network, input=784)

mynet.train(train_dataset, 100, 'least-square', 'gradient-descent', 0.05, 10000, model_path='./', dropout_p=0.5)

print(mynet.query(train_dataset[0][0], model_path='./'), train_dataset[0][1])
