import gzip

import numpy as np

def mnist_data():
    return get_images("train"), get_labels("train"), get_images("test"), get_labels("test")

# Source: https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python
def get_images(type):
    with gzip.open(f'data/{type}-images.gz', 'r') as f:

        magic_number = int.from_bytes(f.read(4), 'big')
        image_count = int.from_bytes(f.read(4), 'big')
        row_count = int.from_bytes(f.read(4), 'big')
        column_count = int.from_bytes(f.read(4), 'big')

        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        return images


def get_labels(type):
    with gzip.open(f'data/{type}-labels.gz', 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        label_count = int.from_bytes(f.read(4), 'big')

        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels