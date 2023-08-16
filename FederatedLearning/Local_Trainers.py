import idx2numpy
import numpy as np
import time
import subprocess
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import os
import Proof_Generator as proof
import Data_Quality_Checks as DQ


#Number of Trainers for our Federated Learning Simulation
trainers_number = 10
local_trainer_img = []
local_trainer_label = []
labels = []
imgs = []
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


def init(batchsize:int):
    #Init Dataset and variables
    file = '../Data/MNIST/raw/train-images-idx3-ubyte'
    file2 =  '../Data/MNIST/raw/train-labels-idx1-ubyte'
    #Set batchsize same as size in label_heterogenity.zok batchsize*traniers_number cannot > 60.000
    batchsize = batchsize
    #heterogenity Index to FAIL Biggest amount vs smallest amount; lower number easier to fail
    minimum = int(0.046 * batchsize)

    #Read MNIST Dataset as Array
    global imgs,labels
    imgs = idx2numpy.convert_from_file(file)
    imgs = imgs.copy()
    labels = idx2numpy.convert_from_file(file2)
    labels = labels.copy()

    #Create Local Trainers 
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

def qualityCheck():    
        #Print DataQuality Stats
    for i in range(len(local_trainer_label)):
        classes = DQ.heterogenity_check(local_trainer_label[i])
        if (max(classes)-min(classes))> 1000:
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
def generateProof(proofname:str, batchsizes,params,imgs,labels):
    path = os.getcwd() + f"/ProofGeneration/{proofname}"
    global local_trainer_label,local_trainer_img
    #proof.zok_compile(path,"image_quality_check_nova", "image_quality_check_nova.r1cs", "pallas")
    times=[]
    for x in batchsizes:
        local_trainer_img[0]=(imgs[0:x])
        local_trainer_label[0]= (labels[0:x])
    #print(witness)
        t1 = time.time()
        proof.zok_prove_nova(local_trainer_img[0],local_trainer_label[0],params[0],params[1], proofname,"0",x)
        t2 = time.time()
        times.append((t2-t1)/60)
        clean = ["sh","cleanup.sh"]
        subprocess.run(clean, capture_output= True)
    return times
    
#ZERO KNOWLEGE PROOF GENERATION
#Executing zokrates to r1cs etc.
#print(local_trainer_img[0])
if __name__ == "__main__":
    init(1000)
    batchsizes = [100,200,500,1000]
    batches = ["100","200","500","1000"]
    proofname = "image_variance_nova"
    path = os.getcwd() + f"/ProofGeneration/{proofname}.zok"
    #proof.zok_compile(path,proofname,proofname+".r1cs","pallas")
    times = generateProof(proofname,batchsizes,[True,True],imgs,labels)

    
    fig, ax = plt.subplots()
    ax.bar(batches,times)
    ax.set_ylabel("time(min)")
    ax.set_xlabel("Batchsize")
    ax.set_title(proofname)
    plt.show()
    plt.savefig(f"{proofname}.png", dpi = fig.dpi)
    
