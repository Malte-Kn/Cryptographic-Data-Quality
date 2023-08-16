


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


#Pairwiese difference and general difference per Class
def imageVariance_check(trainerImgs,trainerLabel):
    #Input General Image for every Label
    pixels = len(trainerImgs[1][1])
    whiteCountavg = [98,48,78,77,61,53,73,66,72,63]
    currwhiteCount = 0
    firstavg = [50,56,49,51,49,43,50,47,48]
    lastavg = [13,14,14,13,14,11,15,13,13,14]
    currfirst = 0
    currlast = 0
    res = [[0,0,0,0]]*10
    for x in range(len(trainerImgs)):
        for i in range(pixels):
            for j in range(pixels):
                if trainerImgs[x][i][j] >= 230:
                    currwhiteCount += 1
                    currlast = j
                    if currfirst == [0]:
                        currfirst = i
        res[trainerLabel[x]][3] += 1
        a = (currwhiteCount - whiteCountavg[trainerLabel[x]])**2
        res[trainerLabel[x]][0] += (a-res[trainerLabel[x]][0])/res[trainerLabel[x]][3]
        b = (currfirst - firstavg[trainerLabel[x]])**2
        res[trainerLabel[x]][1] += (b-res[trainerLabel[x]][1])/res[trainerLabel[x]][3]
        c = (currlast - lastavg[trainerLabel[x]])**2 
        res[trainerLabel[x]][2] += (a-res[trainerLabel[x]][2])/res[trainerLabel[x]][3]
        currlast = 0
        currfirst = 0
        currwhiteCount = 0
        


    return res

def imagestats(imgs,labels):
    whitecount = [0]*10
    firstcount = [0]*10
    lastcount = [0]*10
    currfirst = 0
    currlast = 0
    for x in range(10000):
        for i in range(28):
            for j in range(28):
                if imgs[x][i][j] >=230:
                    whitecount[labels[x]] +=1
                    currlast = j
                    if currfirst == 0:
                        currfirst = i
        firstcount[labels[x]] += currfirst
        lastcount[labels[x]] += currlast
    print(f"Whitecount: {whitecount}\n Firsts: {firstcount}\n Lasts: {lastcount}")
    

#TODO WELL DEFINED