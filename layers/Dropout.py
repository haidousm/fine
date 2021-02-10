import numpy as np


class Dropout:

    def __init__(self, keep):
        self.drop = 1 - keep

    def forward(self, inputs, training):
        if not training:
            self.output = inputs.copy()
            return

        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.drop, size=inputs.shape) / self.drop
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
