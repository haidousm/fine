import numpy as np


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        dbiases = np.sum(dvalues, axis=0, keepdims=True)
        dweights = np.dot(self.inputs.T, dvalues)
        dinputs = np.dot(dvalues, self.weights.T)

        self.dbiases = dbiases
        self.dweights = dweights
        self.dinputs = dinputs

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
