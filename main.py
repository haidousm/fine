import numpy as np
import matplotlib.pyplot as plt
from datasets.MNIST_Data import mnist_data
from layers.Layer_Convolution import Layer_Convolution

X, y, X_test, y_test = mnist_data()
X = np.expand_dims(X, axis=3)

X = X[:10]

conv1 = Layer_Convolution(3, (3, 3), 1, 1)
conv1.forward(X)

image_1 = X[1]
conv_img_1 = conv1.output[1]

plt.imshow(image_1, cmap="gray")
plt.show()

plt.imshow(conv_img_1[0], cmap="gray")
plt.show()
plt.imshow(conv_img_1[1], cmap="gray")
plt.show()
plt.imshow(conv_img_1[2], cmap="gray")
plt.show()
