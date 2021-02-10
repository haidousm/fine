from datasets import load_CIFAR10
from models import Model

from layers import Layer_Convolution
from layers import Layer_MaxPool
from layers import Layer_Flatten
from layers import Layer_Dense

from activation_functions import Activation_ReLU
from activation_functions import Activation_Softmax

from loss_functions import Loss_CategoricalCrossEntropy

from models.model_utils import Accuracy_Categorical

from optimizers import Optimizer_Adam

X_train, y_train, X_val, y_val, X_test, y_test = load_CIFAR10()

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

        Layer_Convolution(64, (32, 3, 3)),
        Activation_ReLU(),
        Layer_Convolution(64, (64, 3, 3)),
        Activation_ReLU(),
        Layer_MaxPool((2, 2)),

        Layer_Flatten(),
        Layer_Dense(1024, 64),
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

model.train(X_train, y_train, epochs=5, batch_size=120, print_every=1)
model.evaluate(X_val, y_val, batch_size=120)
