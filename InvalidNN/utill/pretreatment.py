import numpy as np
import urllib.request
import numpy
import gzip
import os
import struct
# 전처리 (dataset rank = 1)


def one_hot(data_set, range_of_data, value=(0, 1)):
    return [
        [value[1] if i==data else value[0] for i in range(*range_of_data)] for data in data_set
    ]


def data_normalization(data_set, method='min-max', rnge=(0, 1)):
    result = []
    if method == 'min-max':
        max_old = max([max(sample) for sample in data_set])
        min_old = min([min(sample) for sample in data_set])

        data_set = np.array(data_set)
        return  data_set/(max_old-min_old) * (max(rnge)-min(rnge)) + min(rnge)

    elif method == 'z-core':
        pass
    elif method == 'decimal-scaling':
        pass
    else:
        pass
    return result


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
    for f in file.values():
        download(f)

    def load_lbl(file):
        file_path = dataset_dir + "/" + file
        with gzip.open(file_path, 'rb') as f:
            lbls = np.frombuffer(f.read(), np.uint8, offset=8)
        return lbls

    def load_img(file):
        file_path = dataset_dir + "/" + file
        with gzip.open(file_path, 'rb') as f:
            imgs = np.frombuffer(f.read(), np.uint8, offset=16)
        imgs = imgs.reshape(-1, 784)
        return imgs

    train_data = data_normalization(load_img(file['train_img']), rnge=(0.01, 0.99))
    if not flatten:
        train_data = train_data.reshape([-1, 28, 28, 1])
    train_label = one_hot(load_lbl(file['train_lbl']), (0, 10), value=(0.01, 0.99))
    test_data = data_normalization(load_img(file['test_img']), rnge=(0.01, 0.99))
    test_label = one_hot(load_lbl(file['test_lbl']), (0, 10), value=(0.01, 0.99))

    train_dataset = [[train_data[i], train_label[i]] for i in range(len(train_data))]
    test_dataset = [[test_data[i], test_label[i]] for i in range(len(test_data))]

    return train_dataset, test_dataset
