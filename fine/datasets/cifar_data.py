import numpy as np
import pickle
import os
import pathlib
import tarfile

from fine.utils.download_file import download_file

CURRENT_DIRECTORY = pathlib.Path(__file__).parent.absolute()

'''
0: airplane
1: automobile
2: bird
3: cat
4: deer
5: dog
6: frog
7: horse
8: ship
9: truck
'''


def load_pickle(f):
    return pickle.load(f, encoding='latin1')


def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y


def _load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def load_CIFAR10(num_training=49000, num_validation=1000, num_test=10000):
    cifar10_path = f'{CURRENT_DIRECTORY}/data/cifar-10-batches-py'
    data_unzipped = os.path.exists(cifar10_path)
    if not data_unzipped:

        cifar10_gzip_path = f'{CURRENT_DIRECTORY}/data/cifar-10-python.tar.gz'
        data_downloaded = os.path.exists(cifar10_gzip_path)
        if not data_downloaded:

            print("downloading CIFAR10...")
            dir_exists = os.path.exists(f'{CURRENT_DIRECTORY}/data')
            if not dir_exists:
                pathlib.Path(f'{CURRENT_DIRECTORY}/data').mkdir(parents=True, exist_ok=True)
            url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            filename = url.split("/")[-1]
            download_file(f'{CURRENT_DIRECTORY}/data/{filename}', url)

        tar = tarfile.open(cifar10_gzip_path, "r:gz")
        tar.extractall(f'{CURRENT_DIRECTORY}/data/')
        tar.close()
        os.remove(cifar10_gzip_path)

    X_train, y_train, X_test, y_test = _load_CIFAR10(cifar10_path)

    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    return X_train, y_train, X_val, y_val, X_test, y_test
