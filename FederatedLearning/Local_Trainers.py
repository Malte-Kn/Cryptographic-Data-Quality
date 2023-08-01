import idx2numpy
import numpy as np
import time
import subprocess
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import os
import Proof_Generator as proof
import Data_Quality_Checks as DQ


#Functions to Change Dataset e.g. Add Blur/Noise change Labels
def label_flipping(labels):
    for i in range(len(labels)):
        if i%4 == 0:
            if labels[i] == 9:
                labels[i] = 0
            else:
                labels[i] +=1
    return labels

def label_heterogenity(labels):
    for i in range(len(labels)):
        if i%3 == 0:
            if labels[i] == 0:
                labels[i] = 6
    return labels

def add_blur(imgs,frequency):
    for i in range(len(imgs)):
        if i%frequency == 0:
            imgs[i] = gaussian_filter(imgs[i], sigma = 1)
            #Show the Blurred Images
            #plt.imshow(imgs[i], cmap='Greys', interpolation='nearest')
            #plt.show()
    return imgs



#Init Dataset and variables
file = '../Data/MNIST/raw/train-images-idx3-ubyte'
file2 =  '../Data/MNIST/raw/train-labels-idx1-ubyte'
#Set batchsize same as size in label_heterogenity.zok batchsize*traniers_number cannot > 60.000
batchsize = 1000
#Number of Trainers for our Federated Learning Simulation
trainers_number = 10
#heterogenity Index to FAIL Biggest amount vs smallest amount; lower number easier to fail
minimum = int(0.046 * batchsize)

#Read MNIST Dataset as Array
imgs = idx2numpy.convert_from_file(file)
imgs = imgs.copy()
labels = idx2numpy.convert_from_file(file2)
labels = labels.copy()

#Create Local Trainers 
local_trainer_img = []
local_trainer_label = []
for i in range (0, trainers_number*batchsize,batchsize):
    local_trainer_img.append(imgs[i:batchsize+i])
    local_trainer_label.append(labels[i:batchsize+i])

#Artifical hetrogen and label fliping for Trainer 3/7
#print(f"{label_heterogenity(local_trainer_label[3])}")

#Change some 0 Labels to 6 to induce Heterogenity
local_trainer_label[3] = label_heterogenity(local_trainer_label[3])
#Change some Labels to +1 (9 to 0)
local_trainer_label[7] = label_flipping(local_trainer_label[7])

#Add Blur to every Image 
local_trainer_img[5] = add_blur(local_trainer_img[5],1)
local_trainer_img[5] = add_blur(local_trainer_img[5],1)
#plt.imshow(local_trainer_img[5][0], cmap='Greys', interpolation='nearest')
#plt.show()
#Add Blur to every 10th Image 
local_trainer_img[6] = add_blur(local_trainer_img[6],10)
#imgsset = imgs[:batchsize]
#labelsset = labels[:batchsize]
'''
#Print DataQuality Stats
for i in range(len(local_trainer_label)):
    classes = DQ.heterogenity_check(local_trainer_label[i])
    if (max(classes)-min(classes))> minimum:
        print(f"The Dataset of Trainer {i} is not evenly distributed: Digit {classes.index(max(classes))}: {max(classes)} and Digit {classes.index(min(classes))}: {min(classes)}\n")
    badImgs = DQ.imageQuality_check(local_trainer_img[i])
    if badImgs != [] and len(badImgs) < 15:
        print(f"Trainer {i} has (a) bad Image(s): {len(badImgs)}: {badImgs}\n")
    elif badImgs != [] and len(badImgs) >= 15 :
        print(f"Trainer {i} has many bad Images: {len(badImgs)}\n")
    #To Show Bad Images

    #if badImgs != [] and i != 5:
     #   for x in badImgs:
      #      print(f"Trainer {i} has a bad Image: {x}")#local_trainer_img[i][x]
       #     plt.imshow(local_trainer_img[i][x], cmap='Greys', interpolation='nearest')
        #    plt.show()
#print(number_of_labels)
#print(f"{len(local_trainer_label[5])}")

'''
#ZERO KNOWLEGE PROOF GENERATION
#Executing zokrates to r1cs etc.
#print(local_trainer_img[0])
path = os.getcwd() + "/ProofGeneration/label_heterogenity_nova.zok"
proof.zok_compile(path,"label_heterogenity_nova", "label_heterogenity_nova.r1cs", "pallas")

#print(witness)

proof.zok_prove_nova(local_trainer_label[0],False, "label_heterogenity_nova","0",1000)
'''
path = "/home/malte/Desktop/bachelor/Thesisgit/thesis/Implementation/FederatedLearning/LocalTrainers/label_heterogenity.zok"
proof.zok_compile(path,"label_heterogenity", "label_heterogenity.r1cs")
for i in range(len(local_trainer_label)):
    witness = proof.witness_input(local_trainer_label[i]) + str(minimum)
    proof.zok_prove(witness, "label_heterogenity",str(i))
'''