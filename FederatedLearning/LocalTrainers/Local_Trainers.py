import idx2numpy
import numpy as np
import time
import subprocess
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

#Forms data to Format for the witness generatin for ZKproof generation
def witness_input(data):
    input = ""
    for x in range(len(data)):
        input +=  str(data[x]) +" "
    return input

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

#Quality Checks outside of Zero Knowledge
def heterogenity_check(trainerLabel):
    #Class distribution
    classes = [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(trainerLabel)):
        classes[trainerLabel[i]] += 1
    return classes #(max(classes) - min(classes)) < minimum

def imageQuality_check(trainerImgs):
    pixels = len(trainerImgs[1][1])
    badImgs = []
    gray = 0
    distribution = [0,0,0,0,0,0]
    for x in range(len(trainerImgs)):
        for i in range(pixels):
            for j in range(pixels):
                if trainerImgs[x][i][j] < 30:
                    distribution[0] += 1
                elif trainerImgs[x][i][j] < 90:
                    distribution[1] += 1
                elif trainerImgs[x][i][j] < 150:
                    distribution[2] += 1
                elif trainerImgs[x][i][j] < 200:
                    distribution[3] += 1
                elif trainerImgs[x][i][j] < 225:
                    distribution[4] += 1
                elif trainerImgs[x][i][j] <= 255:
                    distribution[5] += 1
        gray = distribution[1] + distribution[2]+distribution[3]+distribution[4]
        if gray > 500 or distribution[5] < 10:
            badImgs.append(x)
        distribution = [0,0,0,0,0,0]
    return badImgs  


#Zokrates ZKP Creation
def zok_compile(path:str, output:str, r1cs:str):
    zokrates_compile = ["zokrates","compile","-i",path, "-o",output,"-r", r1cs]
    print("Compiling root.zok\n")
    t_start = time.time()
    x = subprocess.run(zokrates_compile, capture_output= True)
    t_end= time.time()
    print(f"Compiling took {t_end-t_start} sec")

    zokrates_setup = ["zokrates", "setup", "-i", output, "-p", output+"_proving.key","-v",output+"_verification.key"]
    print("Setup \n")
    t_start = time.time()
    x = subprocess.run(zokrates_setup, capture_output= True)
    t_end= time.time()
    print(f"Setup took {t_end-t_start} sec")

#Witness and Proof Generation
def zok_prove(witness:str, output:str, trainer:str):
    zokrates_witness = ["zokrates", "compute-witness", "-i", output, "-o", output+"_witness", "-a"]
    zokrates_witness.extend(witness.split(" "))
    print(f"Creating Witness for {batchsize} labels, Witness stored in {output}_witness\n")
    t_start = time.time()
    x = subprocess.run(zokrates_witness, capture_output= True)
    t_end= time.time()
    print(f"Witness creation took {t_end-t_start} sec")

    zokrates_proof = ["zokrates", "generate-proof", "-i", output, "-j" , output+"trainer"+trainer+"_proof.json", "-p", output+"_proving.key", "-w", output+"_witness"]
    print(f"Creating Proof for {batchsize} labels of Trainer {trainer}, Proof stored in {output}{trainer}_proof.json \n")
    t_start = time.time()
    x = subprocess.run(zokrates_proof, capture_output= True)
    t_end= time.time()
    print(f"Proof creation took {t_end-t_start} sec")




#Init Dataset and variables
file = '../../Data/MNIST/raw/train-images-idx3-ubyte'
file2 =  '../../Data/MNIST/raw/train-labels-idx1-ubyte'
#Set batchsize same as size in label_heterogenity.zok batchsize*traniers_number cannot > 60.000
batchsize = 1000
#Number of Trainers for our Federated Learning Simulation
trainers_number = 10
#heterogenity Index to FAIL Biggest amount vs smallest amount
minimum = int(0.045 * batchsize)

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
#plt.imshow(local_trainer_img[5][0], cmap='Greys', interpolation='nearest')
#plt.show()
#Add Blur to every 10th Image 
local_trainer_img[6] = add_blur(local_trainer_img[6],10)
#imgsset = imgs[:batchsize]
#labelsset = labels[:batchsize]
'''
#Print DataQuality Stats
for i in range(len(local_trainer_label)):
    classes = heterogenity_check(local_trainer_label[i])
    if (max(classes)-min(classes))> minimum:
        print(f"The Dataset of Trainer {i} is not evenly distributed: Digit {classes.index(max(classes))}: {max(classes)} and Digit {classes.index(min(classes))}: {min(classes)}\n")
    badImgs = imageQuality_check(local_trainer_img[i])
    if badImgs != [] and len(badImgs) < 15:
        print(f"Trainer {i} has (a) bad Image(s): {len(badImgs)}: {badImgs}\n")
    elif badImgs != [] and len(badImgs) >= 15 :
        print(f"Trainer {i} has many bad Images: {len(badImgs)}\n")
    #To Show Bad Images
'''
    #if badImgs != [] and i != 5:
     #   for x in badImgs:
      #      print(f"Trainer {i} has a bad Image: {x}")#local_trainer_img[i][x]
       #     plt.imshow(local_trainer_img[i][x], cmap='Greys', interpolation='nearest')
        #    plt.show()
#print(number_of_labels)
#print(f"{len(local_trainer_label[5])}")

#ZERO KNOWLEGE PROOF GENERATION
#Executing zokrates to r1cs etc.


path = "/home/malte/Desktop/bachelor/Thesisgit/thesis/Implementation/FederatedLearning/LocalTrainers/label_heterogenity.zok"
zok_compile(path,"label_heterogenity", "label_heterogenity.r1cs")
for i in range(len(local_trainer_label)):
    witness = witness_input(local_trainer_label[i]) + str(minimum)
    zok_prove(witness, "label_heterogenity",str(i))

