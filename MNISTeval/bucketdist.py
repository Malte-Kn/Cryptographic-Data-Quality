import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

#Select bucket place for each pixel
def bucket(img:cv2) ->str:
    bucket = [0,0,0,0,0]
    for i in range(28):
        for j in range(28):
            if img[i][j]>230:
                bucket[4] += 1
            elif img[i][j]>170:
                bucket[3] += 1
            elif img[i][j]>90:
                bucket[2] += 1
            elif img[i][j]>5:
                bucket[1] += 1
            else:
                bucket[0] += 1
    return str(bucket)


#Compairing some test Images
img = cv2.imread("./readtestimg.jpg",0)
b30img = cv2.imread("./blurimg.jpg",0)
b35img = cv2.imread("./GaussBlur35.jpg",0)
n12img = cv2.imread("./GaussNoise12.jpg",0)
n17img = cv2.imread("./GaussNoise17.jpg",0)

buckets = bucket(img)
bucketsb30 = bucket(b30img)
bucketsb35 = bucket(b35img)
bucketsn12 = bucket(n12img)
bucketsn17 = bucket(n17img)

print("Image Bucket : "+buckets+"\n")
print("BlurImage (3,7,30) Bucket: "+bucketsb30+"\n")
print("BlurImage (21,21,3.5) Bucket: "+bucketsb35+"\n")
print("NoiseImage (1.2) Bucket: "+bucketsn12+"\n")
print("NoiseImage (1.7) Bucket: "+bucketsn17+"\n")


