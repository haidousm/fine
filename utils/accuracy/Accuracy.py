import numpy as np

class Accuracy:

    def calculate(self, predictions, y):

        comparisons = self.compare(predictions, y)

        accuracy = np.mean(comparisons)

        return accuracy
