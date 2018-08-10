import numpy as np
# 전처리 (dataset rank = 1)


def one_hot(data_set, range_of_data, value=(0, 1)):
    return [
        [value[1] if i==data else value[0] for i in range(*value)] for data in data_set
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
