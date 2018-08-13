from InvalidNN import invalidnn as inv
from InvalidNN.utill import pretreatment
from InvalidNN.utill import test
import csv

training_data = []
x_data = []
y_data = []

f = open("D:\Programming\Dataset\MNIST\mnist_train.csv")
for line in csv.reader(f):
    training_data.append(line)

for sample in training_data:
    x_data.append([int(s) for s in sample[1:]])
    y_data.append(int(sample[0]))
training_data = []
x_data = pretreatment.data_normalization(x_data, 'min-max', (0.01, 0.99))
y_data = pretreatment.one_hot(y_data, (0, 10), value=(0.01, 0.99))
for i in range(len(x_data)):
    training_data.append([x_data[i], y_data[i]])

sample_network = [
    inv.Dense('fc_1', 'sigmoid', 200, dropout=True),
    inv.Dense('fc_2', 'softmax', 10)
]

mynet = inv.NeuralNetwork(sample_network, input=784)

print(mynet.query(training_data[0][0], model_path='./model'))

mynet.train(training_data, 100, 'least-square', 'gradient-descent', 0.05, 10000, model_path='./model')
