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

        self.mask = np.zeros((n_inputs, n_channels, output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + pool_height
                w_start = j * self.stride
                w_end = w_start + pool_width
                input_slice = inputs[:, :, h_start:h_end, w_start:w_end]
                max_index = np.max(input_slice, axis=(2, 3))
                output[:, :, i, j] = max_index
                self.mask[:, :, i, j] = max_index

        self.output = output

    def backward(self, dvalues):
        n_inputs, n_channels, dvalue_height, dvalue_width = dvalues.shape
        pool_height, pool_width = self.pool_size

        dinputs = np.zeros(self.input_shape)

        for i in range(dvalue_height):
            for j in range(dvalue_width):
                x_start = j * self.stride
                x_end = x_start + pool_width

                y_start = i * self.stride
                y_end = y_start + pool_height

                dinputs_slice = dinputs[:, :, y_start:y_end, x_start:x_end]
                dinputs_slice = dinputs_slice.reshape(n_inputs, n_channels,
                                                     pool_height * pool_width)
                idx = self.mask[:, :, i, j].astype(int)

                dinputs_slice[:, :, idx] = dvalues[:, :, i, j]
                dinputs_slice = dinputs_slice.reshape(n_inputs, n_channels,
                                                      pool_height, pool_width)
                dinputs[:, :, y_start:y_end, x_start:x_end] = dinputs_slice

        self.dinputs = dinputs
