from datasets.Spiral_Data import spiral_data
from utils.Model import Model

from layers.Layer_Dense import Layer_Dense
from layers.Layer_Dropout import Layer_Dropout

from activation_functions.Activation_ReLU import Activation_ReLU
from activation_functions.Activation_Softmax import Activation_Softmax

from loss_functions.Loss_CategoricalCrossEntropy import Loss_CategoricalCrossEntropy

from optimizers.Optimizer_SGD import Optimizer_SGD

from utils.accuracy.Accuracy_Categorical import Accuracy_Categorical

X, y = spiral_data(samples=100, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

MoussaNet = Model()

MoussaNet.add(Layer_Dense(2, 64))
MoussaNet.add(Activation_ReLU())
MoussaNet.add(Layer_Dense(64, 3))
MoussaNet.add(Activation_Softmax())

MoussaNet.set(
    loss=Loss_CategoricalCrossEntropy(),
    optimizer=Optimizer_SGD(decay=1e-3, momentum=.9),
    accuracy=Accuracy_Categorical())

MoussaNet.finalize()
MoussaNet.train(
    X, y,
    epochs=10001,
    print_every=100)
