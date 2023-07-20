import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
#some Blur test

img = cv2.imread("./testimage.jpg",0)

blurImg = cv2.blur(img,(5,5))

cv2.imwrite("blurimg.jpg",blurImg)