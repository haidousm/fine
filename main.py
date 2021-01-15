import numpy as np
import matplotlib.pyplot as plt
from datasets.MNIST_Data import mnist_data
from layers.Layer_Convolution import Layer_Convolution
from layers.Layer_Dense import Layer_Dense
from layers.Layer_Flatten import Layer_Flatten
from activation_functions.Activation_ReLU import Activation_ReLU
from activation_functions.Activation_Softmax import Activation_Softmax
from loss_functions.Loss_CategoricalCrossEntropy import Loss_CategoricalCrossEntropy

from optimizers.Optimizer_SGD import Optimizer_SGD

from utils.Model import Model
from utils.accuracy.Accuracy_Categorical import Accuracy_Categorical

X, y, X_test, y_test = mnist_data()
X = np.expand_dims(X, axis=3)

X = np.array([image.astype(np.float32) / 255 for image in X])

X = X[:10000]
y = y[:10000]

model = Model()

model.add(Layer_Convolution(32, (3, 3), 1, 1))
model.add(Activation_ReLU())
model.add(Layer_Convolution(32, (3, 3), 1, 1))
model.add(Activation_ReLU())
model.add(Layer_Flatten())
model.add(Layer_Dense(3920, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_ReLU())
model.add(Activation_Softmax())

model.set(loss=Loss_CategoricalCrossEntropy(),
          optimizer=Optimizer_SGD(),
          accuracy=Accuracy_Categorical())

model.finalize()
model.train(X, y,
            epochs=1,
            batch_size=128,
            print_every=1)

# confidences = model.predict(X_test[:5])
# predictions = model.output_layer_activation.predictions(confidences)
# print(predictions)
# print(y_test[:5])
