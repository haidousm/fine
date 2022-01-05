import numpy as np


class MaxPool2D:

    def __init__(self, pool_size, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs, training):
        n_inputs, n_channels, input_height, input_width = inputs.shape
        pool_height, pool_width = self.pool_size

        inputs_reshaped = inputs.reshape(
            n_inputs, n_channels, input_height // pool_height,
            pool_height, input_width // pool_width, pool_width)

        output = inputs_reshaped.max(axis=3).max(axis=4)

        self.inputs = inputs
        self.inputs_reshaped = inputs_reshaped
        self.output = output

    def backward(self, dvalues):
        inputs = self.inputs
        inputs_reshaped = self.inputs_reshaped

        output = self.output

        dinputs_reshaped = np.zeros_like(inputs_reshaped)
        out_newaxis = output[:, :, :, np.newaxis, :, np.newaxis]

        mask = (inputs_reshaped == out_newaxis)

        dvalues_newaxis = dvalues[:, :, :, np.newaxis, :, np.newaxis]
        dvalues_broadcast, _ = np.broadcast_arrays(dvalues_newaxis, dinputs_reshaped)

        dinputs_reshaped[mask] = dvalues_broadcast[mask]
        dinputs_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)

        dinputs = dinputs_reshaped.reshape(inputs.shape)

        self.dinputs = dinputs
