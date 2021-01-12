import numpy as np

from datasets.Spiral_Data import spiral_data
from datasets.MNIST_Data import mnist_data
from utils.Model import Model

from layers.Layer_Dense import Layer_Dense

from activation_functions.Activation_ReLU import Activation_ReLU
from activation_functions.Activation_Softmax import Activation_Softmax

from loss_functions.Loss_CategoricalCrossEntropy import Loss_CategoricalCrossEntropy

from optimizers.Optimizer_SGD import Optimizer_SGD

from utils.accuracy.Accuracy_Categorical import Accuracy_Categorical

np.set_printoptions(linewidth=200)

X, y, X_test, y_test = mnist_data()

X = np.array([image.astype(np.float32) / 255 for image in X])
X_test = np.array([image.astype(np.float32) / 255 for image in X_test])

X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

model = Model()

model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

model.set(loss=Loss_CategoricalCrossEntropy(),
          optimizer=Optimizer_SGD(decay=1e-3),
          accuracy=Accuracy_Categorical())

model.finalize()
model.train(X, y,
            validation_data=(X_test, y_test),
            epochs=10,
            batch_size=128,
            print_every=100)
