from InvalidNN import InvalidNN as inv
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
print(training_data[0])

sample_network = [
    inv.FullyConnected('sigmoid', 200, dropout=True),
    inv.FullyConnected('softmax', 10)
]

mynet = inv.NeuralNetwork(sample_network, input_units=784)

mynet.train(training_data, 10, 'least-square', 'gradient-descent', 0.05, epoch=150)

print('정확도 : ', test.test_model(mynet, training_data))
