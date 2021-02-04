import gzip
import numpy as np
from utils.augment_image import augment_image


def load_mnist():
    X_train = __get_images("train")
    X_test = __get_images("test")

    y_train = __get_labels("train")
    y_test = __get_labels("test")

    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    return X_train, y_train, X_test, y_test


def load_mnist_augmented():
    X_train_inaug, y_train, X_test_inaug, y_test = load_mnist()

    X_train_aug = np.array([[augment_image(image * 255.0) for image in channel] for channel in X_train_inaug], dtype=np.float32) / 255.0
    X_test_aug = np.array([[augment_image(image * 255.0) for image in channel] for channel in X_test_inaug], dtype=np.float32) / 255.0

    X_train = np.concatenate((X_train_aug, X_train_inaug))
    y_train = np.concatenate((y_train, y_train))

    X_test = np.concatenate((X_test_aug, X_test_inaug))
    y_test = np.concatenate((y_test, y_test))

    keys = np.array(range(X_train.shape[0]))
    np.random.shuffle(keys)

    X_train = X_train[keys]
    y_train = y_train[keys]

    keys = np.array(range(X_test.shape[0]))
    np.random.shuffle(keys)

    X_test = X_test[keys]
    y_test = y_test[keys]

    return X_train, y_train, X_test, y_test


def __get_images(data_type):
    with gzip.open(f'data/{data_type}-images.gz', 'r') as f:
        int.from_bytes(f.read(4), 'big')

        image_count = int.from_bytes(f.read(4), 'big')
        row_count = int.from_bytes(f.read(4), 'big')
        column_count = int.from_bytes(f.read(4), 'big')

        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8) \
            .reshape((image_count, row_count, column_count))
        return images


def __get_labels(data_type):
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
