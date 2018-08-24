# 모델을 입력받고 테스트


def test_model(model, test_data):
    def argmax(x):
        return x.index(max(x))

    return sum([
        argmax(model.query(sample[0])) == argmax(sample[1]) for sample in test_data
    ])/len(test_data)
