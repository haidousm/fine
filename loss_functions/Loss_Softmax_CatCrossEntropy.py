import numpy as np

from activation_functions.Activation_Softmax import Activation_Softmax
from loss_functions.Loss_CategoricalCrossEntropy import Loss_CategoricalCrossEntropy


class Loss_Softmax_CatCrossEntropy:

    def __init__(self):
        self.test = "HI"
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs, training=True)
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
