import numpy as np


class Softmax:

    def forward(self, inputs, training):
        self.inputs = inputs

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        dinputs = np.empty_like(dvalues)
        output = self.output

        for index, (single_output, single_dvalues) in enumerate(zip(output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

        self.dinputs = dinputs

    @staticmethod
    def predictions(output):
        return np.argmax(output, axis=1)
