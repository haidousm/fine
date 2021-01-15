import numpy as np
import matplotlib.pyplot as plt


class Layer_Convolution:

    def __init__(self, n_kernels, kernel_shape, padding, stride,
                 weight_regularizer_l1=0., weight_regularizer_l2=0.,
                 bias_regularizer_l1=0., bias_regularizer_l2=0.):
        self.weights = 0.01 * np.random.randn(n_kernels, *kernel_shape)
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
        n_inputs, n_channels, input_height, input_width, = self.inputs.shape
        n_kernels, _, kernel_height, kernel_width = self.weights.shape

        output_height = int(1 + (input_height + 2 * padding - kernel_height) / stride)
        output_width = int(1 + (input_width + 2 * padding - kernel_width) / stride)

        padded_input = np.pad(inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')

        output = np.zeros((n_inputs, n_kernels, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * stride
                h_end = h_start + kernel_height
                w_start = j * stride
                w_end = w_start + kernel_width

                a = padded_input[:, np.newaxis, :, h_start:h_end, w_start:w_end]
                b = self.weights[np.newaxis, :, :, :, :]
                # print(a.shape)
                # print(b.shape)
                ab = (a * b).sum((2, 3, 4))
                output[:, :, i, j] = ab
                # output[:, :, i, j] = np.sum(
                #     padded_input[:, :, h_start:h_end, w_start:w_end, np.newaxis] *
                #     self.weights[np.newaxis, :, :, :],
                #     axis=(2, 3, 4)
                # )

        self.output = output

    def backward(self, dvalues):

        padding = self.padding
        stride = self.stride

        n_inputs, n_channels, input_height, input_width = self.inputs.shape
        n_kernels, _, kernel_height, kernel_width = self.weights.shape
        _, _, dvalue_height, dvalue_width = dvalues.shape

        self.dbiases = np.sum(dvalues, axis=(0, 2, 3)) / n_inputs

        input_padded = np.pad(self.inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')

        dweights = np.zeros_like(self.weights)
        dinputs = np.zeros_like(input_padded)
        for i in range(dvalue_height):
            for j in range(dvalue_width):
                h_start = i * stride
                h_end = h_start + kernel_height
                w_start = j * stride
                w_end = w_start + kernel_width

                dinputs[:, :, h_start:h_end, w_start:w_end] += np.sum(
                    self.weights[np.newaxis, :, :, :, :] *
                    dvalues[:, :, i:i + 1, j:j + 1, np.newaxis],
                    axis=1
                )

                a = input_padded[:, :, h_start:h_end, w_start:w_end]
                b = dvalues[:, :, i:i + 1, j:j + 1]
                ab = (a * b).sum(axis=0)
                ab = np.expand_dims(ab, axis=1)

                dweights += ab

        self.dweights = dweights / n_inputs
        self.dinputs = dinputs[:, :, padding:padding + input_height, padding:padding + input_width]
        # plt.imshow(self.inputs[0, 0], cmap="gray")
        # plt.show()
