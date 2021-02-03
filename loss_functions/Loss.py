import numpy as np


class Loss:

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        return data_loss

    def calculate_accumulated(self):
        data_loss = self.accumulated_sum / self.accumulated_count

        return data_loss

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
