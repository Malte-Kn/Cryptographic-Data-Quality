import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
#Some noisetests
img = cv2.imread("./testimage.jpg",0)
uni_noise=np.zeros((28,28),dtype=np.uint8)
cv2.randu(uni_noise,0,255)
uni_noise=(uni_noise*0.7).astype(np.uint8)
img = (255-img)
un_img=cv2.add(img,uni_noise)
un_img = 255-un_img
cv2.imwrite("invnoiseimg.jpg",un_img)
un_img=cv2.add(un_img,uni_noise)

cv2.imwrite("noiseimg.jpg",un_img)

