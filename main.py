import numpy as np

from datasets.Spiral_Data import spiral_data
from datasets.MNIST_Data import  mnist_data
from utils.Model import Model

from layers.Layer_Dense import Layer_Dense

from activation_functions.Activation_ReLU import Activation_ReLU
from activation_functions.Activation_Softmax import Activation_Softmax

from loss_functions.Loss_CategoricalCrossEntropy import Loss_CategoricalCrossEntropy

from optimizers.Optimizer_SGD import Optimizer_SGD

from utils.accuracy.Accuracy_Categorical import Accuracy_Categorical

#X, y = spiral_data(samples=100, classes=3)
np.set_printoptions(linewidth=200)

X, y, X_test, y_test = mnist_data()

X = np.array([image.astype(np.float32) / 255 for image in X])
X_test = np.array([image.astype(np.float32) / 255 for image in X_test])

X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

MoussaNet = Model()

MoussaNet.add(Layer_Dense(X.shape[1], 64))
MoussaNet.add(Activation_ReLU())
MoussaNet.add(Layer_Dense(64, 64))
MoussaNet.add(Activation_ReLU())
MoussaNet.add(Layer_Dense(64, 10))
MoussaNet.add(Activation_Softmax())

MoussaNet.set(
    loss=Loss_CategoricalCrossEntropy(),
    optimizer=Optimizer_SGD(decay=1e-3, momentum=.9),
    accuracy=Accuracy_Categorical())

MoussaNet.finalize()

for i in range(11):
    print(f"epoch: {i + 1}")
    for j in range(50):

        batch_X = X[:(j+1)*100]
        batch_y = y[:(j+1)*100]
        MoussaNet.train(
            batch_X, batch_y,
            epochs=1,
            step=j + 1,
            print_every=10)
