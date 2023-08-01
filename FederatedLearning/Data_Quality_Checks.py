


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

#TODO IMAGE DISTANCE; WELL DEFINED