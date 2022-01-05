import numpy as np
import random
import cv2


def augment_image(img):
    x = random.randrange(-5, 5, 2)
    y = random.randrange(-4, 3, 2)
    rows, cols = img.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    augmented_image = cv2.warpAffine(img, M, (cols, rows))
    return augmented_image
