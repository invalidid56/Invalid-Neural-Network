from InvalidNN import invalidnn as inv
from InvalidNN.utill import pretreatment
import tensorflow as tf


train_dataset, test_dataset = pretreatment.mnist_download(dataset_dir='/media/hdd/datasets')

print(next(train_dataset))

print(next(train_dataset))

print(next(train_dataset))