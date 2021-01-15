import numpy as np


class Layer_MaxPooling:

    def __init__(self, pool_size, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs, training):
        self.input_shape = inputs.shape
        n_inputs, n_channels, input_height, input_width = inputs.shape
        pool_height, pool_width = self.pool_size

        output_height = 1 + (input_height - pool_height) // self.stride
        output_width = 1 + (input_width - pool_width) // self.stride

        output = np.zeros((n_inputs, n_channels, output_height, output_width))

        self.mask = np.zeros((n_channels, output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                for c in range(n_channels):
                    x_start = j * self.stride
                    x_end = x_start + pool_width

                    y_start = i * self.stride
                    y_end = y_start + pool_height
                    inputs_slice = inputs[:, c, y_start:y_end, x_start:x_end]

                    output[:, c, i, j] = np.max(inputs_slice, axis=(1, 2))

                    max_index = np.argmax(
                        inputs_slice.reshape(n_inputs, 1, inputs_slice.shape[1] * inputs_slice.shape[2]), axis=2)
                    self.mask[c, i, j] = max_index
        self.output = output

    def backward(self, dvalues):
        n_inputs, n_channels, dvalue_height, dvalue_width = dvalues.shape
        pool_height, pool_width = self.pool_size

        dinputs = np.zeros(self.input_shape)

        for i in range(dvalue_height):
            for j in range(dvalue_width):
                for c in range(n_channels):
                    x_start = j * self.stride
                    x_end = x_start + pool_width

                    y_start = i * self.stride
                    y_end = y_start + pool_height

                    dinputs_slice = dinputs[:, c, y_start:y_end, x_start:x_end]
                    dinputs_slice = dinputs_slice.reshape(n_inputs, 1,
                                                         pool_height * pool_width)
                    idx = int(self.mask[c, i, j])
                    dinputs_slice[:, 0, idx] = dvalues[:, 0, i, j]
                    dinputs_slice = dinputs_slice.reshape(n_inputs, 1,
                                                          pool_height, pool_width)
                    dinputs[:, c, y_start:y_end, x_start:x_end] = dinputs_slice
        self.dinputs = dinputs
