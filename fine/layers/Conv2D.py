import numpy as np
from fine.utils.im2col_cython import im2col_cython, col2im_cython


class Conv2D:

    def __init__(self,
                 n_kernels, kernel_shape,
                 padding=1, stride=1):
        self.weights = 0.01 * np.random.randn(n_kernels, *kernel_shape).astype(np.float32)
        self.biases = np.zeros((n_kernels, 1))

        self.padding = padding
        self.stride = stride

    def forward(self, inputs, training):
        n_inputs, n_channels, input_height, input_width, = inputs.shape
        n_kernels, _, kernel_height, kernel_width = self.weights.shape

        padding = self.padding
        stride = self.stride

        inputs_col = im2col_cython(inputs, kernel_height, kernel_width, padding, stride)
        weights = self.weights.reshape(n_kernels, -1)

        output = weights @ inputs_col + self.biases
        output = output.reshape(n_kernels, input_height, input_width, n_inputs).transpose(3, 0, 1, 2)

        self.inputs = inputs
        self.inputs_col = inputs_col
        self.output = output

    def backward(self, dvalues):
        padding = self.padding
        stride = self.stride

        n_inputs, n_channels, input_height, input_width, = self.inputs.shape
        n_kernels, _, kernel_height, kernel_width = self.weights.shape
        _, _, dvalue_height, dvalue_width = dvalues.shape

        inputs_col = self.inputs_col

        dbiases = np.sum(dvalues, axis=(0, 2, 3)) / n_inputs

        dvalues = dvalues.transpose(1, 2, 3, 0).reshape(n_kernels, -1)
        dweights = dvalues @ inputs_col.T
        dweights = dweights.reshape(self.weights.shape)

        weights = self.weights.reshape(n_kernels, -1)
        dinputs_col = weights.T @ dvalues
        dinputs = col2im_cython(dinputs_col, n_inputs, n_channels, input_height, input_width, kernel_height,
                                kernel_width, padding=padding, stride=stride)

        self.dbiases = dbiases.reshape(n_kernels, -1)
        self.dweights = dweights
        self.dinputs = dinputs

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
