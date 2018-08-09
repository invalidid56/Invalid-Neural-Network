from InvalidNN import InvalidNN as inv
from InvalidNN.utill import pretreatment
import csv

training_data = []
result = []

f = open("D:\Programming\Dataset\MNIST\mnist_train.csv")
for line in csv.reader(f):
    training_data.append(line)

for sample in training_data:
    result.append([[int(s) for s in sample[1:]], int(sample[0])])
training_data = result

training_data = pretreatment.one_hot(training_data, (0, 10), value=(0.01, 0.99))
training_data = pretreatment.data_normalization(training_data, rnge=(0.01, 0.09))

sample_network = [
    inv.FullyConnected('sigmoid', 200, dropout=True),
    inv.FullyConnected('softmax', 10)
]
print(len(training_data))
mynet = inv.NeuralNetwork(sample_network, input_units=784)

mynet.train(training_data, 10, 'least-square', 'gradient-descent', 0.05, epoch=150)

print(mynet.query(training_data[0][0]))
print(training_data[0][1])      