import numpy as np


class Accuracy:

    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)

        accuracy = np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy

    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
