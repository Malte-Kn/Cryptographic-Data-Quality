import idx2numpy
import numpy as np
import time
import subprocess
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import os,glob
import Proof_Generator as proof
import Data_Quality_Checks as DQ
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import sys
import shutil

#Number of Trainers for our Federated Learning Simulation
trainers_number = 3
batch = 100
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
        if i%2 == 0 or i%3 == 0:
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

def changeResolution(imgs, batch,size):

    transform = T.Resize(size = (size,size))
    res = []
    for i in range(batch):
        img = Image.fromarray(imgs[i])
        img = transform(img)
        x = np.asarray(img)
        res.append(x)
    return res
    
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
    if trainers_number >= 3:
        local_trainer_label[2] = label_heterogenity(local_trainer_label[2])
    #Change some Labels to +1 (9 to 0)
    if trainers_number >= 5:
        local_trainer_label[4] = label_flipping(local_trainer_label[4])

    #Add Blur to every Image 
    if trainers_number >= 4:
        local_trainer_img[3] = add_blur(local_trainer_img[3],1)
        local_trainer_img[3] = add_blur(local_trainer_img[3],1)
    #plt.imshow(local_trainer_img[5][0], cmap='Greys', interpolation='nearest')
    #plt.show()
    #Add Blur to every 10th Image 
    local_trainer_img[1] = add_blur(local_trainer_img[1],2)
    #imgsset = imgs[:batchsize]
    #labelsset = labels[:batchsize]
def qualityCheck():    
        #Print DataQuality Stats
    for i in range(len(local_trainer_label)):
        classes = DQ.heterogenity_check(local_trainer_label[i])
        print(f"The Dataset distribution of Trainer {i} is: Max Digit {classes.index(max(classes))}: {max(classes)} and Min Digit {classes.index(min(classes))}: {min(classes)}\n")
        badImgs = DQ.imageQuality_check(local_trainer_img[i])
        
        if badImgs != [] and len(badImgs) < 15:
            print(f"Trainer {i} has (a) bad Image(s): {len(badImgs)}/{len(local_trainer_img[i])}: {badImgs}\n")
        elif badImgs != [] and len(badImgs) >= 15 :
            print(f"Trainer {i} has many bad Images: {len(badImgs)}/{len(local_trainer_img[i])}\n")
        imgVar = DQ.imageVariance_check(local_trainer_img[i],local_trainer_label[i])
        #print(imgVar)
        #To Show Bad Images

        #if badImgs != [] and i != 5:
        #   for x in badImgs:
        #      print(f"Trainer {i} has a bad Image: {x}")#local_trainer_img[i][x]
        #     plt.imshow(local_trainer_img[i][x], cmap='Greys', interpolation='nearest')
            #    plt.show()
    #print(number_of_labels)
    #print(f"{len(local_trainer_label[5])}")
def generateProof(proofname:str, batchsizes,params,imgs,labels,init,trainer):
    path = os.getcwd() + f"/ProofGeneration/{proofname}"
    global local_trainer_label,local_trainer_img
    #proof.zok_compile(path,"image_quality_check_nova", "image_quality_check_nova.r1cs", "pallas")
    times=[]
    stats=[]
    for x in batchsizes:
        local_trainer_img[0]=(imgs[0:x])
        local_trainer_label[0]= (labels[0:x])
    #print(witness)
        t1 = time.time()
        y = proof.zok_prove_nova(local_trainer_img[0],local_trainer_label[0],params[0],params[1], proofname,trainer,x,init)
        t2 = time.time()
        times.append((t2-t1))
        stats.append(os.stat("Proof"+proofname+"trainer"+trainer+"_proof.json").st_size/(1024*1024))
        print(y)
        clean = ["sh","cleanup.sh"]
        subprocess.run(clean, capture_output= True)
    return times,stats

def continueProof(proofname:str, batchsizes,params,imgs,labels,init):
    global local_trainer_label,local_trainer_img
    src = "1024"+proofname[:-4]+"trainer"+"0"+"_proof.json"
    dest = proofname+"trainer"+"0"+"_proof.json"
    if os.path.exists(dest):
        os.remove(dest)
    shutil.copy(src,dest)
    #proof.zok_compile(path,"image_quality_check_nova", "image_quality_check_nova.r1cs", "pallas")
    times=[]
    stats=[]
    for x in batchsizes:
        local_trainer_img[0]=(imgs[0:x])
        local_trainer_label[0]= (labels[0:x])
    #print(witness)
        t1 = time.time()
        proof.zok_continue_nova(local_trainer_img[0],local_trainer_label[0],params[0],params[1], proofname,"0",x,init)
        t2 = time.time()
        times.append((t2-t1))
        stats.append(os.stat(proofname+"trainer"+"0"+"_proof.json").st_size/(1024*1024))
        clean = ["sh","cleanup.sh"]
        subprocess.run(clean, capture_output= True)
    return times,stats
def compressproof(proofname:str,params,batchsizes,init):
    times=[]
    times2=[]
    
    for x in batchsizes:
    #print(witness)
        input=str(x)+proofname[:-4]+"trainer0_proof.json"
        y,z = proof.zok_compress_nova(proofname,input,params[0],params[1],x,init)
        times.append(y)
        times2.append(z)
        #stats.append(os.stat(proofname+"trainer"+"0"+"_proof.json").st_size/(1024*1024))
        print(y)
        clean = ["sh","cleanup.sh"]
        subprocess.run(clean, capture_output= True)
    return times,times2
#ZERO KNOWLEGE PROOF GENERATION
#Executing zokrates to r1cs etc.
#print(local_trainer_img[0])
def main():
    init(batch)
    batchsizes = [batch]
    #proofnames = ["image_quality_check30"]
    proofnames = ["image_quality_check_nova","image_variance_nova","image_welldef_check_nova","label_heterogenity_nova"]
    inits = [["0","0"],[["0","0","0","0"]]*10,["0","0","0"],["0","0","0","0","0","0","0","0","0","0"]]
    params = [[True,False],[True,True],[True,True],[False,True]]
    qualityCheck()

    for i in range(0,trainers_number):
        for j in range(0,len(proofnames)):
            path = os.getcwd() + f"/ProofGeneration/{proofnames[j]}.zok"
            #proofnames[j] = proofnames[j]+str(sys.argv[1])
            timescs,statscs = proof.zok_compile(path,proofnames[j],proofnames[j]+".r1cs","pallas")
            times,stats = generateProof(proofnames[j],batchsizes,params[j],local_trainer_img[i],labels,inits[j],str(i))
            for filename in glob.glob("./"+proofnames[j]+"*"):
                os.remove(filename)
    print("\n-----Completed------")
    '''For Different Measurments
    proofnames = ["image_quality_check_nova_7","image_quality_check_nova_14","image_quality_check_nova_21","image_quality_check_nova","image_variance_nova","image_welldef_check_nova","combined_quality_check_nova","label_heterogenity_nova"]
    inits = [["0","0"],["0","0"],["0","0"],["0","0"],[["0","0","0","0"]]*10,["0","0","0"],([["0","0","0","0","0"]]*10,["0","0"]),["0","0","0","0","0","0","0","0","0","0"]]
    params = [[True,False],[True,False],[True,False],[True,False],[True,True],[True,True],[True,True],[False,True]]
    temp = local_trainer_img[5]
    for j in range(1,len(proofnames)):
        path = os.getcwd() + f"/ProofGeneration/{proofnames[j]}.zok"
        proofnames[j] = proofnames[j]+str(sys.argv[1])
        timescs,statscs = proof.zok_compile(path,proofnames[j],proofnames[j]+".r1cs","pallas")
        #Input for Proofs: Name,Batches, Image,Label Inputs both or one
        if j == 1:
            local_trainer_img[5] = changeResolution(local_trainer_img[0],1050,14)
        elif j == 2:
            local_trainer_img[5] = changeResolution(local_trainer_img[0],1050,21)
        elif j == 0:
            local_trainer_img[5] = changeResolution(local_trainer_img[0],1050,7)

        times,stats = generateProof(proofnames[j],batchsizes,params[j],local_trainer_img[5],labels,inits[j])
        #times,stats = continueProof(proofnames[j],batchsizes,params[j],local_trainer_img[0],labels,inits[j])
        #timescom,timesver = compressproof(proofnames[j],params[j],batchsizes,inits[j])
        if times[0] > 5:
            with open("measurements"+proofnames[j],"a") as f:
                f.write(proofnames[j]+"\n")
                f.write("Compiletime\n")
                f.write(str(timescs)+"\n")                
                f.write("Proofgeneration times\n")
                f.write(str(times)+"\n")
                f.write("Compilesize,Setupsize\n")
                f.write(str(statscs)+"\n")
                f.write("Proofsizes\n")
                f.write(str(stats)+"\n")                
                
        local_trainer_img[0]=temp
        for filename in glob.glob("./"+proofnames[j]+"*"):
            os.remove(filename)
    '''    

if __name__ == "__main__":
    main()