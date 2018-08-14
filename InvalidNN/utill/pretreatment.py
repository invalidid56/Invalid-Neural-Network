import numpy as np
import mnist
import numpy
# 전처리 (dataset rank = 1)


def one_hot(data_set, range_of_data, value=(0, 1)):
    return [
        [value[1] if i==data else value[0] for i in range(*range_of_data)] for data in data_set
    ]


def data_normalization(data_set, method='min-max', rnge=(0, 1)):
    result = []
    if method == 'min-max':
        if isinstance(data_set[0], numpy.ndarray):
            data_set = [sample.flatten() for sample in data_set]
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


def mnist_pretreatment():
    train_img = data_normalization(mnist.train_images(), rnge=(0.01, 0.99))
    train_lbl = one_hot(mnist.train_labels(), (0, 10), value=(0.01, 0.99))
    train_dataset = [[train_img[i], train_lbl[i]] for i in range(60000)]
    test_img = data_normalization(mnist.test_images(), rnge=(0.01, 0.99))
    test_lbl = one_hot(mnist.test_labels(), (0, 10), value=(0.01, 0.99))
    test_dataset = [[test_img[i], test_lbl[i]] for i in range(10000)]
    return train_dataset, test_dataset

