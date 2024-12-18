import os
import time
import subprocess
import json

#Forms data to Format for the witness generatin for ZKproof generation
def witness_input_label(data):
    input = ""
    for x in range(len(data)):
        input +=  str(data[x]) +" "
    return input


def witness_input_label_nova(data):
    input = "["
    for x in range(len(data)):
        input +=  "\""+str(data[x]) +"\""+","
    input = input[:-1]
    input += "]"
    return input


def witness_input_img(data):
    input = ""
    for x in range(len(data)):
        for y in range(len(data[x])):
            for z in range(len(data[x])):
                input +=  str(data[x][y][z]) +" "
    return input
def witness_input_img_nova(data):
    input="["
    for x in range(len(data)):
        input += "["
        for i in range(len(data[0])):
            input+= "["
            for j in range(len(data[0])): 
                if j < len(data[0])-1: 
                    input+= "\"" + str(data[x][i][j])+"\"" + ", "
                else:
                    input+= "\"" + str(data[x][i][j])+"\"" 
            if i < len(data[0])-1:
                input += "],\n"
            else: 
                input += "]\n"
        if x < len(data)-1:        
            input += "],\n"
        else:
            input += "]\n"
    input += "]"
    return input
def witness_input_img_label_nova(imgs,labels):
    input="["
    for x in range(len(imgs)):
        input += "[["
        for i in range(len(imgs[0])):
            input+= "["
            for j in range(len(imgs[0])): 
                if j < len(imgs[0])-1: 
                    input+= "\"" + str(imgs[x][i][j])+"\"" + ", "
                else:
                    input+= "\"" + str(imgs[x][i][j])+"\"" 
            if i < len(imgs[0])-1:
                input += "],\n"
            else: 
                input += "]\n"
        if x < len(imgs)-1:        
            input += f"],\"{labels[x]}\"],\n"
        else:
            input += f" ],\"{labels[x]}\"] \n" 
    input += "]"
    return input
#Zokrates ZKP Creation
def zok_compile(path:str, output:str, r1cs:str, curve: str):
    zokrates_compile = ["zokrates","compile","-i",path, "-o",output,"-r", r1cs, "-c", curve]
    times = []
    stats = []
    print(f"Compiling {output}.zok\n")
    t_start = time.time()
    x = subprocess.run(zokrates_compile, capture_output= True)
    t_end= time.time()
    print(f"Compiling took {t_end-t_start} sec {x}")
    stats.append(os.stat(output+".r1cs").st_size/(1024*1024))
    times.append((t_end-t_start))
    #Only do normal setup if not Nova/Pallas-curve
    if curve != "pallas":
        zokrates_setup = ["zokrates", "setup", "-i", output, "-p", output+"_proving.key","-v",output+"_verification.key"]
        print("Setup \n")
        t_start = time.time()
        x = subprocess.run(zokrates_setup, capture_output= True)
        t_end= time.time()
        print(f"Setup took {t_end-t_start} sec")
    else:
        zokrates_setup = ["zokrates","nova", "setup", "-i", output, "-o", output+".params"]
        print("Setup \n")
        t_start = time.time()
        #Setup onlyruns on certain Systems
        x = subprocess.run(zokrates_setup, capture_output= True)
        t_end= time.time()
        print(f"Setup took {t_end-t_start} sec")
    stats.append(os.stat(output+".params").st_size/(1024*1024))
    times.append((t_end-t_start))
    return times,stats
#Witness and Proof Generation
def zok_prove(input, isimg:bool, output:str, trainer:str, batchsize:int):
    if isimg:
        witness = witness_input_img(input)
    else:
        witness = witness_input_label(input)
    print(f"Creating Witness for {batchsize} items, Witness stored in {output}_witness\n")
    t_start = time.time()
    os.system("zokrates compute-witness -i "+ output + " -o " + output +"_witness" + " -a " + witness)
    t_end= time.time()
    print(f"Witness creation took {t_end-t_start} sec")

    zokrates_proof = ["zokrates", "generate-proof", "-i", output, "-j" , output+"trainer"+trainer+"_proof.json", "-p", output+"_proving.key", "-w", output+"_witness"]
    print(f"Creating Proof for {batchsize} items of Trainer {trainer}, Proof stored in {output}{trainer}_proof.json \n")
    t_start = time.time()
    x = subprocess.run(zokrates_proof, capture_output= True)
    t_end= time.time()
    print(f"Proof creation took {t_end-t_start} sec")


def zok_prove_nova(input,input2, isimg:bool,islabel:bool, output:str, trainer: str, batchsize:int,init):
    print(f"Creating Witness for {batchsize} items, Witness stored in {output}steps.json\n")
    if isimg and not islabel:
        #init = [0,0]
        f = open (f"{output}init.json", "w")
        f.write(witness_input_label_nova(init))
        f.close()
        g = open (f"{output}steps.json", "w")
        g.write(witness_input_img_nova(input))
        g.close()
    elif not isimg and islabel:
        #init = [0]*10
        f = open (f"{output}init.json", "w")
        f.write(witness_input_label_nova(init))
        f.close()
        g = open (f"{output}steps.json", "w")
        g.write(witness_input_label_nova(input2))
        g.close()
    else:
        #init = [["0","0","0","0"]]*10
        #init=[0,0,0]
        f = open (f"{output}init.json", "w")
        f.write(json.dumps(init))
        f.close()
        g = open (f"{output}steps.json", "w")
        g.write(witness_input_img_label_nova(input,input2))
        g.close()
    zokrates_proof = ["zokrates", "nova", "prove", "-i", output, "-j" , "Proof"+output+"trainer"+trainer+"_proof.json", "-p", output+".params", "--init", output+"init.json", "--steps", output+"steps.json"]
    #print(zokrates_proof)
    print(f"Creating Proof for {batchsize} items of Trainer {trainer}, Proof stored in Proof{output}trainer{trainer}_proof.json \n")
    t_start = time.time()
    x = subprocess.run(zokrates_proof, capture_output= True)
    t_end= time.time()
    print(f"Proof creation took {t_end-t_start} sec")
    return x

def zok_continue_nova(input,input2, isimg:bool,islabel:bool, output:str, trainer: str, batchsize:int,init):
    print(f"Creating Witness for {batchsize} items, Witness stored in {output}steps.json\n")
    if isimg and not islabel:
        #init = [0,0]
        f = open (f"{output}init.json", "w")
        f.write(witness_input_label_nova(init))
        f.close()
        g = open (f"{output}steps.json", "w")
        g.write(witness_input_img_nova(input))
        g.close()
    elif not isimg and islabel:
        #init = [0]*10
        f = open (f"{output}init.json", "w")
        f.write(witness_input_label_nova(init))
        f.close()
        g = open (f"{output}steps.json", "w")
        g.write(witness_input_label_nova(input2))
        g.close()
    else:
        #init = [["0","0","0","0"]]*10
        f = open (f"{output}init.json", "w")
        f.write(json.dumps(init))
        f.close()
        g = open (f"{output}steps.json", "w")
        g.write(witness_input_img_label_nova(input,input2))
        g.close()
    zokrates_proof = ["zokrates", "nova", "prove", "-c", "-i", output, "-j" , output+"trainer"+trainer+"_proof.json", "-p", output+".params", "--init", output+"init.json", "--steps", output+"steps.json"]
    print(f"Creating Proof for {batchsize} items of Trainer {trainer}, Proof stored in {output}trainer{trainer}_proof.json \n")
    t_start = time.time()
    x = subprocess.run(zokrates_proof, capture_output= True)
    t_end= time.time()
    print(f"Proof creation took {t_end-t_start} sec")
    return x

def zok_compress_nova(proofname,input,isimg:bool,islabel:bool,batchsize,init):
    if isimg and not islabel:
        #init = [0,0]
        f = open (f"{proofname}init.json", "w")
        f.write(witness_input_label_nova(init))
        f.close()
        
    elif not isimg and islabel:
        #init = [0]*10
        f = open (f"{proofname}init.json", "w")
        f.write(witness_input_label_nova(init))
        f.close()
        
    else:
        #init = [["0","0","0","0"]]*10
        #init=[0,0,0]
        f = open (f"{proofname}init.json", "w")
        f.write(json.dumps(init))
        f.close()
    zokrates_proof = ["zokrates", "nova", "compress", "--i", input, "-p", proofname+".params","-j" , str(batchsize)+proofname+"COMproof.json","-v", proofname+".key"]
    print(zokrates_proof)
    print(f"Crompress Proof for {batchsize} items of Trainer 0, Proof stored in {proofname}trainer0_proof.json \n")
    t_start = time.time()
    x = subprocess.run(zokrates_proof, capture_output= True)
    t_end= time.time()
    t1 =t_end-t_start
    print(f"Comprssion took {t_end-t_start} sec")
    zokrates_proof = ["zokrates", "nova", "verify", "-j", str(batchsize)+proofname, "--instance-path" , str(batchsize)+proofname+"trainer0_proof.json", "-v", proofname+".key", "--init", proofname+"init.json"]
    print(f"Verify {batchsize} items of Trainer 0, Proof stored in {proofname}trainer0_proof.json \n")
    t_start = time.time()
    y = subprocess.run(zokrates_proof, capture_output= True)
    t_end= time.time()
    t2 =t_end-t_start
    return t1,t2