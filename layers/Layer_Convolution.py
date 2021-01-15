import numpy as np
import matplotlib.pyplot as plt

class Layer_Convolution:

    def __init__(self, n_kernels, kernel_shape, padding, stride,
                 weight_regularizer_l1=0., weight_regularizer_l2=0.,
                 bias_regularizer_l1=0., bias_regularizer_l2=0.):
        self.weights = 0.01 * np.random.randn(n_kernels, *kernel_shape, 1)
        self.biases = 0.01 * np.random.randn(n_kernels)
        self.padding = padding
        self.stride = stride

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):

        self.inputs = inputs
        padding = self.padding
        stride = self.stride
        n_inputs, input_height, input_width, n_channels = self.inputs.shape
        n_kernels, kernel_height, kernel_width, _ = self.weights.shape

        output_height = int(1 + (input_height + 2 * padding - kernel_height) / stride)
        output_width = int(1 + (input_width + 2 * padding - kernel_width) / stride)

        padded_input = np.pad(inputs, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')

        output = np.zeros((n_inputs, n_kernels, output_height, output_width))

        for inp_index in range(n_inputs):
            for kern_index in range(n_kernels):
                for i in range(output_height):
                    for j in range(output_width):
                        for c in range(n_channels):
                            output[inp_index, kern_index, i, j] = (self.weights[kern_index] * padded_input[inp_index,
                                                                                              i: i + kernel_height,
                                                                                              j: j + kernel_width,
                                                                                              c]).sum() + self.biases[
                                                                      kern_index]

        self.output = output

    def backward(self, dvalues):

        padding = self.padding
        stride = self.stride

        n_inputs, input_height, input_width, n_channels = self.inputs.shape
        n_kernels, kernel_height, kernel_width, _ = self.weights.shape
        _, _, dvalue_height, dvalue_width = dvalues.shape

        self.dbiases = np.sum(dvalues, axis=(0, 2, 3))

        input_padded = np.pad(self.inputs, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')

        dweights_height = int(1 + (input_height + 2 * padding - dvalue_height) / stride)
        dweights_width = int(1 + (input_width + 2 * padding - dvalue_width) / stride)

        dweights = np.zeros((n_kernels, dweights_height, dweights_width, n_channels))
        for inp_index in range(n_inputs):
            for kern_index in range(n_kernels):
                for i in range(dweights_height):
                    for j in range(dweights_width):
                        for c in range(n_channels):
                            dweights[kern_index, i, j, c] = (
                                    dvalues[inp_index, kern_index] + input_padded[inp_index,
                                                                     i: i + dvalue_height,
                                                                     j: j + dvalue_width,
                                                                     c]).sum()
        self.dweights = dweights

        dinputs = np.zeros_like(self.inputs)
        dinputs_padded = np.pad(dinputs, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')

        dvalues_padded = np.pad(dvalues, ((0, 0), (0, 0), (kernel_width-1, kernel_width-1), (kernel_height - 1, kernel_height-1 )), 'constant')

        weights_inverted = np.zeros_like(self.weights)
        for i in range(kernel_height):
            for j in range(kernel_width):
                weights_inverted[:, i, j, :] = self.weights[:, kernel_height - i - 1, kernel_width - i - 1, :]

        for inp_index in range(n_inputs):
            for kern_index in range(n_kernels):
                for i in range(input_height + 2 * padding):
                    for j in range(input_width + 2 * padding):
                        for k in range(kernel_height):
                            for l in range(kernel_width):
                                for c in range(n_channels):
                                    dinputs_padded[inp_index, i, j, c] += \
                                        dvalues_padded[inp_index, kern_index, i + k, j + l] *\
                                        weights_inverted[kern_index, k, l, c]

        dinputs = dinputs_padded[:, padding:-padding, padding:-padding, :]
        self.dinputs = dinputs
        print(self.dinputs[0])