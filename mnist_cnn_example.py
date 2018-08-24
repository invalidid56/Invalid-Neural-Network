from InvalidNN import invalidnn as inn
from InvalidNN.utill import pretreatment

training_data, testing_data = pretreatment.mnist_download(flatten=False)

layers = [
    inn.Conv2D('Conv_1', activate_fn='relu', filters=32, filter_shape=[5, 5], stride=[1, 1], padding='same'),
    inn.Pooling('Pooling_1', pooling='max', stride=[2, 2], size=[2, 2], padding='same'),
    inn.Conv2D('Conv_2', 'relu', 64, [5, 5], [1, 1], 'same'),
    inn.Pooling('Pooling_2', 'max', [2, 2], [2, 2], 'same'),
    inn.Dense('Dense', 'relu', 1024, dropout=True),
    inn.Dense('Softmax', 'softmax', 10)
]

mnist_convNet = inn.NeuralNetwork(layers, [28, 28, 1])

mnist_convNet.train(
    train_data=training_data,
    batch_size=100,
    loss_function='cross-entropy',
    optimizer='adam',
    learning_rate=1e-4,
    epoch=5000,
    dropout_p=0.5,
    model_path='./'
)

print(mnist_convNet.query(training_data[0][0]), training_data[0][1])



