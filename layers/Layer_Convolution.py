import numpy as np
from utils.im2col_cython import im2col_cython, col2im_cython

class Layer_Convolution:

    def __init__(self, n_kernels, kernel_shape, padding, stride):
        self.weights = 0.01 * np.random.randn(n_kernels, *kernel_shape).astype(np.float32)
        self.biases = 0.01 * np.random.randn(n_kernels, 1)
        self.dbiases = 0.01 * np.random.randn(n_kernels)
        self.padding = padding
        self.stride = stride

        self.weight_regularizer_l1 = 0
        self.weight_regularizer_l2 = 0
        self.bias_regularizer_l1 = 0
        self.bias_regularizer_l2 = 0

    def forward(self, inputs, training):
        self.inputs = inputs

        n_inputs, n_channels, input_height, input_width, = inputs.shape
        n_kernels, _, kernel_height, kernel_width = self.weights.shape

        padding = self.padding
        stride = self.stride

        inputs = im2col_cython(inputs, kernel_height, kernel_width, padding, stride)
        weights = self.weights.reshape(n_kernels, -1)
        output = weights @ inputs + self.biases

        self.inputs_col = inputs
        self.output = output.reshape(n_kernels, input_height, input_width, n_inputs).transpose(3, 0, 1, 2)

    def backward(self, dvalues):
        padding = self.padding
        stride = self.stride

        n_inputs, n_channels, input_height, input_width, = self.inputs.shape
        n_kernels, _, kernel_height, kernel_width = self.weights.shape
        _, _, dvalue_height, dvalue_width = dvalues.shape

        inputs_col = self.inputs_col

        dbiases = np.sum(dvalues, axis=(0, 2, 3))

        dvalues = dvalues.transpose(1, 2, 3, 0).reshape(n_kernels, -1)
        dweights = dvalues @ inputs_col.T
        dweights = dweights.reshape(self.weights.shape)

        weights = self.weights.reshape(n_kernels, -1)
        dinputs_col = weights.T @ dvalues
        dinputs = col2im_cython(dinputs_col, n_inputs, n_channels, input_height, input_width, kernel_height, kernel_width, padding=padding, stride=stride)

        self.dbiases = dbiases.reshape(n_kernels, -1)
        self.dweights = dweights
        self.dinputs = dinputs
