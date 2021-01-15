import numpy as np
import matplotlib.pyplot as plt
from datasets.MNIST_Data import mnist_data
from layers.Layer_Convolution import Layer_Convolution
from layers.Layer_MaxPooling import Layer_MaxPooling
from layers.Layer_Dropout import Layer_Dropout
from layers.Layer_Flatten import Layer_Flatten
from layers.Layer_Dense import Layer_Dense
from activation_functions.Activation_ReLU import Activation_ReLU
from activation_functions.Activation_Softmax import Activation_Softmax
from loss_functions.Loss_CategoricalCrossEntropy import Loss_CategoricalCrossEntropy

from optimizers.Optimizer_SGD import Optimizer_SGD

from utils.Model import Model
from utils.accuracy.Accuracy_Categorical import Accuracy_Categorical

X, y, X_test, y_test = mnist_data()
X = np.expand_dims(X, axis=1)

X = np.array([image.astype(np.float32) / 255 for image in X])

X = X[:1000]
y = y[:1000]

model = Model()

# (:, 1, 28, 28)
model.add(Layer_Convolution(32, (1, 3, 3), 1, 1))
model.add(Activation_ReLU())

# (:, 32, 28, 28)
model.add(Layer_Convolution(32, (32, 3, 3), 1, 1))
model.add(Activation_ReLU())

# (:, 32, 28, 28)
model.add(Layer_MaxPooling((2, 2)))

# (:, 32, 14, 14)
model.add(Layer_Dropout(0.75))

# (:, 32, 14, 14)
model.add(Layer_Flatten())

# (:, 6272)
model.add(Layer_Dense(6272, 256))
model.add(Activation_ReLU())
model.add(Layer_Dense(256, 32))
model.add(Activation_ReLU())
model.add(Layer_Dense(32, 10))
model.add(Activation_Softmax())

model.set(loss=Loss_CategoricalCrossEntropy(),
          optimizer=Optimizer_SGD(),
          accuracy=Accuracy_Categorical())

model.finalize()
model.train(X, y,
            epochs=1,
            batch_size=128,
            print_every=1)


