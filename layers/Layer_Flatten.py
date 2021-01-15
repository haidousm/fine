import numpy as np

class Layer_Flatten:

    def forward(self, inputs, training):
        self.shape = inputs.shape
        self.output = np.reshape(inputs, (inputs.shape[0], -1))

    def backward(self, dvalues):
        self.dinputs = np.reshape(dvalues, self.shape)
