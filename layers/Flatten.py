import numpy as np


class Flatten:

    def forward(self, inputs, training):
        self.input_shape = inputs.shape
        self.output = np.reshape(inputs, (inputs.shape[0], -1))

    def backward(self, dvalues):
        self.dinputs = np.reshape(dvalues, self.input_shape)
