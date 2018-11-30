<!-- $theme: default -->

Invalid Neural Network: InvalidNN v 0.2 edited by nonameP

===

What Is This?
---
<em>[tensorflow](https://www.tensorflow.org/)</em>에서 신경망을 **빠르게** 작성하고, **편하게** 테스트할 수 있도록 만든 패키지입니다.

How can I Install This? <!-- Installation -->
---
```bash
$ git clone https://github.com/InvalidId404/Invalid-Neural-Network
$ cd Invalid-Neural-Network
```

1. Github의 본 저장소를 클론합니다.

```bash
$ pip3 install numpy tensorflow
```

2. 라이브러리(`numpy`와 `tensorflow`)를 설치합니다. `requirements.txt`를 이용해서 설치할 수도 있습니다.

```python
$ python3
>>> from InvalidNN import InvalidNN as inv
>>>
```

3. 설치가 완료되었습니다. 이제 InvalidNN를 사용하실 수 있습니다!

How can I Use This? <!-- Example -->
---
```Python
from InvalidNN import InvalidNN  # InvalidNN 모듈을 import 합니다

foo = InvalidNN.FullyConnected('sigmoid', 200) # 시그모이드 함수를 사용하고 200개의 유닛을 갖는 전결합층 레이어를 정의합니다.
bar = InvalidNN.FullyConnected('softmax', 10)  # 소프트맥스 함수를 사용하고 10개의 유닛을 갖는 전결합층 레이어.

MyFirstNetwork = InvalidNN.NeuralNetwork([foo, bar], input_units = 784)  # 정의한 두 레이어를 연결하고, 784개의 입력 유닛을 갖는 신경망을 정의합니다.

MyFisrtNetwork.train(training_data, batch_size = 10, loss_function = 'least-loss', optimizer = 'gradient-descent',learning_rate = 0.05, epoch = 150)	
# training data를 학습 데이터셋으로, 배치 사이즈를 10으로, 오차함수를 least-loss로, 옵티마이저를 gradient-descent를, 학습률을 0.05로 하여 신경망을 150번 학습시킵니다

print(MyFisrtNetwork.query(test_data[0][0])) # 테스트 데이터의 첫번째 입력값을 신경망에 질의하고 출력합니다.
```
