import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from utils.ImageGenerator import ImageGenerator

from datasets.Spiral_Data import spiral_data
from datasets.MNIST_Data import mnist_data
from utils.Model import Model

from layers.Layer_Dense import Layer_Dense

from activation_functions.Activation_ReLU import Activation_ReLU
from activation_functions.Activation_Softmax import Activation_Softmax

from loss_functions.Loss_CategoricalCrossEntropy import Loss_CategoricalCrossEntropy

from optimizers.Optimizer_SGD import Optimizer_SGD

from utils.accuracy.Accuracy_Categorical import Accuracy_Categorical

np.set_printoptions(linewidth=200)
X, y, X_test, y_test = mnist_data()
print(X[0].tolist())

X = np.array([image.astype(np.float32) / 255 for image in X])
X_test = np.array([image.astype(np.float32) / 255 for image in X_test])

X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

MoussaNet = Model.load("trained_models/digits_mnist.model")

canvas = tk.Tk()
canvas.wm_geometry("%dx%d+%d+%d" % (400, 400, 10, 10))
canvas.config(bg='white')
image_gen = ImageGenerator(canvas, 10, 10)

while True:
    if image_gen.is_new_image:

        image_data = 255 - np.array(image_gen.image)
        image_data = image_data.reshape(1, -1).astype(np.float32) / 255
        # plt.imshow(image_data.reshape(28, 28))
        # plt.show()
        confidences = MoussaNet.predict(image_data)
        prediction = MoussaNet.output_layer_activation.predictions(confidences)
        image_gen.prediction.set(f"Prediction: {prediction}")
        image_gen.is_new_image = False
    canvas.update_idletasks()
    canvas.update()
