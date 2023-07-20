import idx2numpy
import numpy as np
import time
import subprocess
#Read MNIST Dataset as Array
#Forms data to Format for the witness generatin
def witness_input(data):
    input = ""
    for x in range(len(data)):
        input +=  str(data[x]) +" "
    return input


file = '../../Data/MNIST/raw/train-images-idx3-ubyte'
file2 =  '../../Data/MNIST/raw/train-labels-idx1-ubyte'
#Set batchsize same as size in root.zok
batchsize = 1000
#min of labels of a kind to fail
min = 0
imgs = idx2numpy.convert_from_file(file)
labels = idx2numpy.convert_from_file(file2)
imgsset = imgs[:batchsize]
labelsset = labels[:batchsize]

number_of_labels = [0,0,0,0,0,0,0,0,0,0]
for label in labelsset:
    number_of_labels[label] +=1
#print(number_of_labels)

#Executing zokrates to r1cs etc.
zok = "zokrates"
zokrates_compile = [zok,"compile","-i","/home/malte/Desktop/bachelor/Thesis git/thesis/Implementation/Zokratestest/MNISTlabeltest/root.zok"]
zokrates_setup = [zok, "setup"]
witness = witness_input(labelsset) + str(min)
zokrates_witness = [zok, "compute-witness","-a"]
zokrates_proof = [zok, "generate-proof"]
zokrates_witness.extend(witness.split(" "))

print("Compiling root.zok\n")
t_start = time.time()
x = subprocess.run(zokrates_compile, capture_output= True)
t_end= time.time()
print(f"Compiling took {t_end-t_start} sec")

print("Setup \n")
t_start = time.time()
x = subprocess.run(zokrates_setup, capture_output= True)
t_end= time.time()
print(f"Setup took {t_end-t_start} sec")

print(f"Creating Witness for {batchsize} labels, Witness stored in out.wtn\n")
t_start = time.time()
x = subprocess.run(zokrates_witness, capture_output= True)
t_end= time.time()
print(f"Witness creation took {t_end-t_start} sec")

print(f"Creating Proof for {batchsize} labels, Proof stored in proof.json \n")
t_start = time.time()
x = subprocess.run(zokrates_proof, capture_output= True)
t_end= time.time()
print(f"Proof creation took {t_end-t_start} sec")

