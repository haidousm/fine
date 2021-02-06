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
#
# model = Model(
#     layers=[
#         Layer_Convolution(16, (3, 3, 3)),
#         Activation_ReLU(),
#         Layer_Convolution(16, (16, 3, 3)),
#         Activation_ReLU(),
#         Layer_MaxPool((2, 2)),
#
#         Layer_Convolution(32, (16, 3, 3)),
#         Activation_ReLU(),
#         Layer_Convolution(32, (32, 3, 3)),
#         Activation_ReLU(),
#         Layer_MaxPool((2, 2)),
#
#         Layer_Flatten(),
#         Layer_Dense(2048, 64),
#         Activation_ReLU(),
#         Layer_Dense(64, 64),
#         Activation_ReLU(),
#         Layer_Dense(64, 10),
#         Activation_Softmax()
#     ],
#     loss=Loss_CategoricalCrossEntropy(),
#     optimizer=Optimizer_Adam(decay=1e-3),
#     accuracy=Accuracy_Categorical()
# )
#
# model.finalize()
# model.train(X_train, y_train, epochs=10, batch_size=120, print_every=1)

# model.load_parameters("trained_models/cifar-10.params")
# validation, acc: 0.504, loss: 1.380 time: 76.96s - 10 epochs
# validation, acc: 0.530, loss: 1.313 time: 79.62s - 20 epochs
# validation, acc: 0.561, loss: 1.229 time: 83.02s - 30 epochs
# validation, acc: 0.578, loss: 1.198 time: 88.12s - 40 epochs
# validation, acc: 0.590, loss: 1.171 time: 79.83s - 70 epochs
'''
git filter-branch --force --index-filter \
  "git rm -r --cached --ignore-unmatch venv/" \
  --prune-empty --tag-name-filter cat -- --all

'''