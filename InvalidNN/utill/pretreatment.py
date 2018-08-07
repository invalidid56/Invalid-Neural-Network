# 전처리 (dataset은 [인풋, 레이블])


def one_hot(data_set, range_of_data, value=(0, 1)):
    return [
            [
                sample[0], [value[1] if i != sample[-1] else value[1] for i in range(*range_of_data)]
            ] for sample in data_set
        ]


def data_normalization(data_set, method='min-max', rnge=(0, 1)):
    result = []
    if method == 'min-max':
        max_old = max([max(sample[0]) for sample in data_set])
        min_old = min([min(sample[0]) for sample in data_set])
        result = [
            [
                [s/(max_old-min_old) * (max(rnge)-min(rnge)) + min(rnge) for s in sample[0]], sample[-1]
            ] for sample in data_set
        ]

    elif method == 'z-core':
        pass
    elif method == 'decimal-scaling':
        pass
    else:
        pass
    return result


def data_cleaning(data_set):
    pass
