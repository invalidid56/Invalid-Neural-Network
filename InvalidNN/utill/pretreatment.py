import numpy as np
import urllib.request
import struct
import numpy
import gzip
import os
import struct
import pickle
import tarfile
# 전처리 (dataset rank = 1)


def one_hot(data, range_of_data, value=(0, 1)):
    return [value[1] if i==data else value[0] for i in range(*range_of_data)]


def data_normalization(sample, method='min-max', range_of_data=(0, 1), new_range=(0, 1)):
    if method == 'min-max':
        return (np.array(sample))/(range_of_data[1]-range_of_data[0]) * (new_range[1]-new_range[0]) + new_range[0]
    elif method == 'z-core':
        pass
    elif method == 'decimal-scaling':
        pass
    else:
        pass


def data_cleaning(data_set):
    pass


def mnist_download(dataset_dir='.', flatten=True):
    download_url = 'http://yann.lecun.com/exdb/mnist/'
    file = {
        'train_img': 'train-images-idx3-ubyte.gz',
        'train_lbl': 'train-labels-idx1-ubyte.gz',
        'test_img': 't10k-images-idx3-ubyte.gz',
        'test_lbl': 't10k-labels-idx1-ubyte.gz'
    }

    def download(file):
        filedir = dataset_dir+'/'+file
        if os.path.exists(filedir):
            return False
        else:
            urllib.request.urlretrieve(download_url + file, filedir)
            return True

    def make_gen(file_x, file_y):
        file_x = os.path.join(dataset_dir, file_x)
        file_y = os.path.join(dataset_dir, file_y)
        with gzip.open(file_x, 'rb') as fx:
            zero, data_type, dims = struct.unpack('>HBB', fx.read(4))
            fx_shape = tuple(struct.unpack('>I', fx.read(4))[0] for d in range(dims))
            with gzip.open(file_y, 'rb') as fy:
                fy.read(8)
                for i in range(fx_shape[0]):
                    data = data_normalization(np.fromstring(fx.read(fx_shape[1]*fx_shape[2]), dtype=np.uint8),
                                              range_of_data=(0, 256), new_range=(0.01, 0.99))
                    label = one_hot(np.fromstring(fy.read(1), dtype=np.uint8)[0], (0, 9), value=(0.01, 0.99))
                    yield (data, label)

    for f in file.values():
        download(f)

    return make_gen(file['train_img'], file['train_lbl']), make_gen(file['test_img'], file['test_lbl'])


def cifar10_download(dataset_dir='.'):
    download_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    file_name = 'cifar-10-python.tar.gz'
    file_path = dataset_dir + '/' + file_name

    if not os.path.exists(file_path):
        urllib.request.urlretrieve(download_url, file_path)

    ext = [ 'cifar-10-batches-py/data_batch_1',
            'cifar-10-batches-py/data_batch_2',
            'cifar-10-batches-py/data_batch_3',
            'cifar-10-batches-py/data_batch_4',
            'cifar-10-batches-py/data_batch_5',
            'cifar-10-batches-py/test_batch']

    dataset = []
    with tarfile.open(file_path, 'r:gz') as tar:
        for file in tar.getnames():
            if file in ext:
                with tar.extractfile(file) as f:
                    dataset.append(pickle.load(f, encoding='bytes'))

    train_data = []
    test_data = []
    for batch in dataset:
        data = np.reshape(data_normalization(batch[b'data'], rnge=(0.01, 0.99)), [-1, 32, 32, 3])
        label = one_hot(batch[b'labels'], (0, 10), value=(0.01, 0.99))
        if batch[b'batch_label'] == b'testing batch 1 of 1':
            test_data += [[data[i], label[i]] for i in range(len(data))]
        else:
            train_data += [[data[i], label[i]] for i in range(len(data))]

    return train_data, test_data

