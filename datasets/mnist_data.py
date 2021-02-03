import gzip
import numpy as np


def load_mnist():
    return get_images("train"), get_labels("train"), get_images("test"), get_labels("test")


def get_images(data_type):
    with gzip.open(f'data/{data_type}-images.gz', 'r') as f:
        int.from_bytes(f.read(4), 'big')

        image_count = int.from_bytes(f.read(4), 'big')
        row_count = int.from_bytes(f.read(4), 'big')
        column_count = int.from_bytes(f.read(4), 'big')

        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8) \
            .reshape((image_count, row_count, column_count))
        return images


def get_labels(data_type):
    with gzip.open(f'data/{data_type}-labels.gz', 'r') as f:
        int.from_bytes(f.read(4), 'big')
        int.from_bytes(f.read(4), 'big')

        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels

# MNIST Digits
# validation, acc: 0.974, loss: 0.111 - Conv v1 - 1 filter - 5 epochs
# validation, acc: 0.961, loss: 0.138 - Conv v2 - 64 filters - 5 epochs
# validation, acc: 0.969, loss: 0.095 - Conv v3 - 64 filters augmented - 3 epochs
# validation, acc: 0.981, loss: 0.060 - Conv v4 - 128 filters augmented - 5 epochs
