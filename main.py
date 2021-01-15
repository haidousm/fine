import numpy as np
import matplotlib.pyplot as plt
from datasets.MNIST_Data import mnist_data
from layers.Layer_Convolution import Layer_Convolution
from layers.Layer_Dense import Layer_Dense
from activation_functions.Activation_ReLU import Activation_ReLU
from loss_functions.Loss_Softmax_CatCrossEntropy import Loss_Softmax_CatCrossEntropy

from optimizers.Optimizer_SGD import Optimizer_SGD

from utils.Model import Model

X, y, X_test, y_test = mnist_data()
X = np.expand_dims(X, axis=3)

X = np.array([image.astype(np.float32) / 255 for image in X])

X = X[:10000]
y = y[:10000]

model = Model()

model.add(Layer_Convolution(1, (3, 3), 1, 1))
model.add(Activation_ReLU())
# relu1 =
# dense1 = Layer_Dense(784, 10)
# loss_softmax = Loss_Softmax_CatCrossEntropy()
#
# optimizer = Optimizer_SGD()
#
# for epoch in range(1):
#     conv1.forward(X)
#     relu1.forward(conv1.output, training=True)
#     flattened_input = np.reshape(relu1.output, (relu1.output.shape[0], -1))
#     dense1.forward(flattened_input, training=True)
#     loss = loss_softmax.forward(dense1.output, y)
#
#     predictions = np.argmax(loss_softmax.output, axis=1)
#     if len(y.shape) == 2:
#         y = np.argmax(y, axis=1)
#     accuracy = np.mean(predictions == y)
#
#     print(f'loss: {loss}, acc: {accuracy}')
#
#     loss_softmax.backward(loss_softmax.output, y)
#     dense1.backward(loss_softmax.dinputs)
#     flattened_dinputs = np.reshape(dense1.dinputs, (dense1.dinputs.shape[0], 1, 28, 28))
#     relu1.backward(flattened_dinputs)
#     conv1.backward(relu1.dinputs)
#
#     optimizer.update_params(conv1)
#     optimizer.update_params(dense1)
#     # plt.imshow(conv1.weights[0], cmap="gray")
#     # plt.show()
#     # print(conv1.dweights[0])
#     # print(dense1.dweights[0])
