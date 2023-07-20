import idx2numpy
import numpy as np
import cv2
#Read MNIST Dataset as Array

file = '../data/MNIST/raw/train-images-idx3-ubyte'
arr = idx2numpy.convert_from_file(file)
# arr is now a np.ndarray type of object of shape 60000, 28, 28

cv2.imwrite("readtestimg.jpg",arr[7])