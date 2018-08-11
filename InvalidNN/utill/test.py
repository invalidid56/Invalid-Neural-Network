# 모델을 입력받고 테스트


def test_model(model, test_data):
    score = 0
    argmax = lambda x: x.index(max(x))
    for sample in test_data:
        output = model.query(sample[0])
        if argmax(output.tolist()) == argmax(sample[1]):
            score += 1
    return score/500
