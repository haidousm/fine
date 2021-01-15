import numpy as np


class Layer_Convolution:
    def __init__(self, n_kernels, kernel_shape, padding, stride):
        self.weights = 0.01 * np.random.randn(n_kernels, *kernel_shape)
        self.biases = 0.01 * np.random.randn(n_kernels)
        self.padding = padding
        self.stride = stride

    def conv2D(self, image, kernel, bias, padding=0, stride=1):
        kernel = np.flipud(np.fliplr(kernel))

        kernel_width, kernel_length = kernel.shape
        image_width, image_length = image.shape

        output_x = int(((image_width - kernel_width + 2 * padding) / stride) + 1)
        output_y = int(((image_length - kernel_length + 2 * padding) / stride) + 1)
        output = np.zeros((output_x, output_y))

        image_padded = np.pad(image,
                              ((padding, padding), (padding, padding)),
                              mode='constant')

        for y in range(image_length):
            if y > image_length - kernel_length:
                break
            if y % stride == 0:
                for x in range(image_width):
                    if x > image_width - kernel_width:
                        break
                    if x % stride == 0:
                        output[x, y] = (kernel * image_padded[x: x + kernel_width, y: y + kernel_length]).sum()

        return output + bias

    def forward(self, inputs):

        self.inputs = inputs
        padding = self.padding
        stride = self.stride
        n_inputs, input_height, input_width, n_channels = self.inputs.shape
        n_kernels, kernel_height, kernel_width = self.weights.shape

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

        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dweights = np.zeros_like(self.weights)
        self.dinputs = np.zeros_like(self.inputs)

        inputs_padded = np.pad(input, ((0,), (0,), (self.padding,), (self.padding,)), 'constant')

        for n in range(self.inputs.shape[0]):
            for f in range(self.weights.shape[0]):
                for i in range(self.weights.shape[1]):
                    for j in range(self.weights.shape[2]):
                        for k in range(dvalues.shape[1]):
                            for l in range(dvalues.shape[2]):
                                for c in range(self.inputs.shape[3]):
                                    self.dweights[f, c, i, j] = inputs_padded[
                                                                    n, c, self.stride * i + k, self.stride * j + l] * \
                                                                self.dinputs[n, f, k, l]

        print(self.dweights)
