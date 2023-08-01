import os
import time
import subprocess

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
        input += "[["
        for i in range(28):
            input+= "["
            for j in range(28): 
                if j < 27: 
                    input+= "\"" + str(data[x][i][j])+"\"" + ", "
                else:
                    input+= "\"" + str(data[x][i][j])+"\"" 
            if i < 27:
                input += "],\n"
            else: 
                input += "]\n"
        if x < len(data)-1:        
            input += "]],\n"
        else:
            input += "]]\n"
    input += "]"
    return input

#Zokrates ZKP Creation
def zok_compile(path:str, output:str, r1cs:str, curve: str):
    zokrates_compile = ["zokrates","compile","-i",path, "-o",output,"-r", r1cs, "-c", curve]
    print(f"Compiling {output}.zok\n{zokrates_compile}")
    t_start = time.time()
    x = subprocess.run(zokrates_compile, capture_output= True)
    t_end= time.time()
    print(f"Compiling took {t_end-t_start} sec")
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
        x = subprocess.run(zokrates_setup, capture_output= True)
        t_end= time.time()
        print(f"Setup took {t_end-t_start} sec")
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


def zok_prove_nova(input, isimg:bool, output:str, trainer: str, batchsize:int):
    print(f"Creating Witness for {batchsize} items, Witness stored in {output}steps.json\n")
    if isimg:
        g = open (f"{output}steps.json", "w")
        g.write(witness_input_img_nova(input))
        g.close()
    else:
        g = open (f"{output}steps.json", "w")
        g.write(witness_input_label_nova(input))
        g.close()
    zokrates_proof = ["zokrates", "nova", "prove", "-i", output, "-j" , output+"trainer"+trainer+"_proof.json", "-p", output+".params", "--init", output+"init.json", "--steps", output+"steps.json"]
    print(f"Creating Proof for {batchsize} items of Trainer {trainer}, Proof stored in {output}trainer{trainer}_proof.json \n{zokrates_proof}")
    t_start = time.time()
    x = subprocess.run(zokrates_proof, capture_output= True)
    t_end= time.time()
    print(f"Proof creation took {t_end-t_start} sec")