import cv2
import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as transf
#Different Quality on MNIST Dataset via tensor and torchvision.transforms

#from AddGaussianNoise https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
transform=transforms.Compose([
        #blur intensity based on sigma, random number between first and second more blur the higher
        transforms.GaussianBlur(kernel_size=(21, 21), sigma=(3.5, 3.5)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        #noise intensity based on input, random number between first and second more blur the higher
        #AddGaussianNoise(1.6, 1.7),
        #need image format
        #transf.adjust_contrast(??img??,0.3)
        #no big effect
        #transforms.RandomAdjustSharpness(400, p=0.5)
        #I think needs colors not just black-white
        #transforms.ColorJitter(brightness=0, contrast=500, saturation=0, hue=0)
        ])
dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)

# Convert the tensor to a numpy array
numpy_image = dataset1[0][0].numpy()

# Convert the numpy array to a cv2 image
cv2_image = np.transpose(numpy_image, (1, 2, 0))
#cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

cv2_image = np.clip(cv2_image * 255, 0, 255)
cv2_image = cv2_image.astype(np.uint8)
# Display the image using cv2
cv2.imwrite("GaussBlur35.jpg", cv2_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()