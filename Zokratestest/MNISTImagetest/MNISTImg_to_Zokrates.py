import idx2numpy
import numpy as np
import time
import subprocess

#Forms data to Format for the witness generatin
def witness_input(imgs):
    input = ""
    for n in range(len(imgs)):
        for x in range(len(imgs[0])):
            for y in range(len(imgs[0])):
                input +=  str(imgs[n][x][y]) + " "
    return input

def witness_input2(data):
    input = ""
    for x in range(len(data)):
        input +=  str(data[x]) +" "
    input = input[:-1]
    return input

def countWhite(imgs):
    res = []
    for x in range(len(imgs)):
        res.append(0)
        for i in range(len(imgs[0])):
            for j in range(len(imgs[0])):
                if imgs[x][i][j] > 240:
                    res[x] += 1
    return res
#Read MNIST Dataset as Array
file = '../../Data/MNIST/raw/train-images-idx3-ubyte'
file2 =  '../../Data/MNIST/raw/train-labels-idx1-ubyte'
#Set batchsize same as size in root.zok
batchsize = 3
#min of labels of a kind to fail
min = 10
imgs = idx2numpy.convert_from_file(file)
labels = idx2numpy.convert_from_file(file2)
imgsset = imgs[:batchsize]
labelsset = labels[:batchsize]
image_to_check = imgs[0]
whiteCount = countWhite(imgsset)
for i in range(len(whiteCount)):
    if whiteCount[i] <= 10:
        print("Not enoght white pixels") 
#Executing zokrates to r1cs etc.

zok = "zokrates"
zokrates_compile = [zok,"compile","-i","/home/malte/Desktop/bachelor/Thesis git/thesis/Implementation/Zokratestest/MNISTImagetest/root.zok"]
zokrates_setup = [zok, "setup"]
witness = witness_input(imgsset) +str(min)
zokrates_witness = [zok, "compute-witness","-a"]
zokrates_proof = [zok, "generate-proof"]
zokrates_witness.extend(witness.split(" "))

print("Compiling root.zok wait\n")
t_start = time.time()
x = subprocess.run(zokrates_compile, capture_output= True)
t_end= time.time()
print(f"Compiling took {t_end-t_start} sec")

print("Setup wait\n")
t_start = time.time()
x = subprocess.run(zokrates_setup, capture_output= True)
t_end= time.time()
print(f"Setup took {t_end-t_start} sec")


print(f"Creating Witness for {whiteCount} White Pixels in Image, Witness stored in witness/out.wtn if not it failed\n")
t_start = time.time()
x = subprocess.run(zokrates_witness, capture_output= True)
t_end= time.time()
print(f"Witness creation took {t_end-t_start} sec")

print(f"Creating Proof for {whiteCount} White Pixels in Image with witness, Proof stored in proof.json. If not it failed\n")
t_start = time.time()
x = subprocess.run(zokrates_proof, capture_output= True)
t_end= time.time()
print(f"Proof creation took {t_end-t_start} sec")

