import numpy as np


class Layer_MaxPooling:

    def __init__(self, pool_size, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}

    def forward(self, inputs, training):
        self.inputs = inputs
        n_inputs, n_channels, input_height, input_width = inputs.shape
        pool_height, pool_width = self.pool_size
        output_height = 1 + (input_height - pool_height) // self.stride
        output_width = 1 + (input_height - pool_width) // self.stride
        output = np.zeros((n_inputs, n_channels, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + pool_height
                w_start = j * self.stride
                w_end = w_start + pool_width
                input_slice = inputs[:, :, h_start:h_end, w_start:w_end]
                self.save_mask(input_slice=input_slice, cords=(i, j))
                output[:, :, i, j] = np.max(input_slice, axis=(2, 3))

        self.output = output

    def backward(self, dvalues):
        dinputs = np.zeros_like(self.inputs)
        _, _, output_height, output_width = dvalues.shape
        pool_height, pool_width = self.pool_size

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + pool_height
                w_start = j * self.stride
                w_end = w_start + pool_width
                dinputs[:, :, h_start:h_end, w_start:w_end] += \
                    dvalues[:, :, i:i + 1, j:j + 1] * self.cache[(i, j)]
        self.dinputs = dinputs

    def save_mask(self, input_slice, cords):
        mask = np.zeros_like(input_slice)
        n_inputs, n_channels, slice_height, slice_width = input_slice.shape
        input_slice = input_slice.reshape(n_inputs, n_channels, slice_height * slice_width)
        idx = np.argmax(input_slice, axis=2)

        n_idx, c_idx = np.indices((n_inputs, n_channels))
        mask.reshape(n_inputs, n_channels, slice_height * slice_width)[n_idx, c_idx, idx] = 1
        self.cache[cords] = mask
