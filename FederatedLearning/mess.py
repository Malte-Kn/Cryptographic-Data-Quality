import ast 
#script to Evaluate multiple Runs into an avarage 
#Quality check; cutoff to disregard outliners;start of first task Id;Numberof Runs
check = "measurementsimage_welldef_check2_nova"
cutoff = 100000
start= 4496
runs = 30
def main():
    mes = []
    mes2 = []
    mesint = []
    mesint2 = []
    mes3 = []
    mes4= []
    mesint3=[]
    mesint4=[]
    res = []
    for i in range(runs):
        with open(f"{check}{start+i}") as f:
            lines = f.readlines()
            for i,x in enumerate(lines):
                if i == 2:
                    mes2.append(lines[i])
                elif i == 4:
                    mes.append(lines[i])
                elif i ==6:
                     mes3.append(lines[i])
                elif i == 8:
                     mes4.append(lines[i])

    for j,list in enumerate(mes):
        mesint.append(ast.literal_eval(list))

    for j2,list2 in enumerate(mes2):
        mesint2.append(ast.literal_eval(list2))
    for j2,list3 in enumerate(mes3):
        mesint3.append(ast.literal_eval(list3))
    for j2,list4 in enumerate(mes4):
        mesint4.append(ast.literal_eval(list4))
    temp = 0
    count = 0
    r = True
    for j in range(len(mesint[0])):
        for x in range(len(mesint)):
            if mesint[x][len(mesint[0])-1]< cutoff:
                temp += mesint[x][j]
                if r:
                    count +=1
        r = False
        temp = temp/count
        res.append(temp)
        temp = 0
    temp = 0
    res2=[]
    for j in range(len(mesint2[0])):
        for x in range(len(mesint2)):
            temp += mesint2[x][j]
        temp = temp/len(mesint2)
        res2.append(temp)
        temp = 0
    res3=[]
    for j in range(len(mesint3[0])):
        for x in range(len(mesint3)):
            temp += mesint3[x][j]
        temp = temp/len(mesint3)
        res3.append(temp)
        temp = 0
    res4=[]
    for j in range(len(mesint4[0])):
        for x in range(len(mesint4)):
            temp += mesint4[x][j]
        temp = temp/len(mesint4)
        res4.append(temp)
        temp = 0
    with open(f"avrg{check}","a") as f:
                    f.write(f"{check}\n")
                    f.write("Compiletime,Setuptime\n")
                    f.write(f"{res2}\n")
                    f.write("Proofgeneration times\n")
                    f.write(f"{res}\n")
                    f.write("Compile/Setupsize\n")
                    f.write(f"{res3}\n")
                    f.write("Proofsize\n")
                    f.write(f"{res4}\n")
                    f.write(f"{count}/10")

if __name__ =="__main__":
    main()
