# def augment_image(img):
#     x = random.randrange(-5, 5, 2)
#     y = random.randrange(-4, 3, 2)
#     rows, cols = img.shape
#     M = np.float32([[1, 0, x], [0, 1, y]])
#     dst = cv2.warpAffine(img, M, (cols, rows))
#     return dst