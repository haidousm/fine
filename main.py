import numpy as np

from datasets import cifar_data
from datasets import mnist_data
from models.Model import Model

from layers.Layer_Convolution import Layer_Convolution
from layers.Layer_Dense import Layer_Dense
from layers.Layer_Flatten import Layer_Flatten
from layers.Layer_MaxPool import Layer_MaxPool

from activation_functions.Activation_ReLU import Activation_ReLU
from activation_functions.Activation_Softmax import Activation_Softmax

from loss_functions.Loss_CategoricalCrossEntropy import Loss_CategoricalCrossEntropy

from models.model_utils.accuracy.Accuracy_Categorical import Accuracy_Categorical

from optimizers.Optimizer_Adam import Optimizer_Adam

X_train, y_train, X_test, y_test = mnist_data.load_mnist()

model = Model(
    layers=[
        Layer_Convolution(16, (3, 3, 3)),
        Activation_ReLU(),
        Layer_Convolution(16, (16, 3, 3)),
        Activation_ReLU(),
        Layer_MaxPool((2, 2)),

        Layer_Convolution(32, (16, 3, 3)),
        Activation_ReLU(),
        Layer_Convolution(32, (32, 3, 3)),
        Activation_ReLU(),
        Layer_MaxPool((2, 2)),

        Layer_Flatten(),
        Layer_Dense(2048, 64),
        Activation_ReLU(),
        Layer_Dense(64, 64),
        Activation_ReLU(),
        Layer_Dense(64, 10),
        Activation_Softmax()
    ],
    loss=Loss_CategoricalCrossEntropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)

model.train(X_train, y_train, epochs=10, batch_size=120)
