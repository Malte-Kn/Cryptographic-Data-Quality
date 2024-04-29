import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast

def main():
    title = "Nova Proof Generation for Data Quality Checks"
    title2 ="Nova Proof Generation for Image Quality\n with different Resolution"
    title3 = "Nova Proof Continuation for Data Quality Checks"
    batches = ['64', '128', '256', '512', '1024']
    proofnames = ['img_quality', 'img_variance', 'img_combined', 'welldef_check', 'label_heterogenity']
    proofnames2 = ["Res28x28","Res21x21","Res14x14","Res10x10"]
    rows2 = ["Res28x28 ","Res21x21","Res14x14","Res10x10"]
    rows=["Quality","Variance","Combined","Well Defined","Heterogenity"]
    rows1=["Quality","Variance","Combined","Well Defined","Label\nHeterogenity"]
    measurements = ["image_quality_check_nova","image_variance_nova","combined_quality_check_nova","image_welldef_check_nova","label_heterogenity_nova"]
    measurements2=["image_quality_check_nova","image_quality_check_nova_21","image_quality_check_nova_14","image_quality_check_nova_10"]
    img_quality = [264,437,700,1504,2882]
    img_quality_stat = [1010231]
    img_quality_stat2 = [110,1132,800,1010231,784]
    img_quality14 = [76,127,233,449,887]
    img_quality14_stat =[6,180,235,257006,196]
    img_quality10_stat =[5,99,112,123445,100]
    img_quality21 =[150,255,463,899,1719]
    img_quality21_stat =[16,360,464,575739,441]
    img_variance = [418,665,1179,2185,4091]
    img_variance_stat = [1420152]
    img_combined = [420,700,1200,2200,4000]
    img_combined_stat = [1569462]
    welldef_check = [350,727,1265,2334,4491]
    welldef_check_stat = [1450231]
    label_heterogenity = [12,19,35,70,138]
    label_heterogenity_stat =[865]
    avrgtimes = []
    avrgtimes2= []
    avrgstats = []
    proofstats=[img_quality_stat,img_variance_stat,img_combined_stat,welldef_check_stat,label_heterogenity_stat]
    for i,x in enumerate(measurements):        
        with open(f"avrgmeasurements{x}") as f:
            lines = f.readlines()
            avrgtimes.append(ast.literal_eval(lines[4]))
            a = ast.literal_eval(lines[7])
            b=ast.literal_eval(lines[9])
            avrgstats.append([round(i)for i in ast.literal_eval(lines[2])]+proofstats[i]+[round(a[1])]+[round(b[0])])
            
    
    for i,x in enumerate(measurements2):        
        with open(f"avrgmeasurements{x}") as f:
            lines = f.readlines()
            avrgtimes2.append(ast.literal_eval(lines[4]))


    prooftimes = [img_quality,img_variance,img_combined,welldef_check,label_heterogenity]
    prooftimes2=[img_quality,img_quality21,img_quality14]
    
    proofstats2=[img_quality_stat2,img_quality21_stat,img_quality14_stat,img_quality10_stat]
    #createPlot_table2(title2,batches,rows2,rows2,avrgtimes2,proofstats2)
    #createPlot_table1CON(title3,batches,rows1,rows,avrgtimes,avrgstats)
    createPlot_table1(title,batches,rows1,rows,avrgtimes,avrgstats)
    
    
    
def createPlot_tableold(title,labels,labels2,rowNames,prooftimes,proofstats):
    Medium = 12
    Big = 14
    colors = ["#738ede","#ff7e42","#62e359","#cf5555","#d584f5"]

    prooftimes = np.asarray(prooftimes)

    
    specs = np.array(proofstats)
    #cLabels = ("Compile Time(s)","Setup Parameter(Mb)","Constrains")
    cLabels = ("Compile Time(s)","Setup Parameter(Mb)","Constrains","Pixel")
    fig, ax1 = plt.subplots(nrows=2)
    plt.rc("font",size=Medium)
    plt.rc("axes",titlesize=Medium)
    plt.rc('axes', labelsize=Medium)    
    plt.rc('xtick', labelsize=Medium)    
    plt.rc('ytick', labelsize=Medium)    
    plt.rc('legend', fontsize=Medium)    
    plt.rc('figure', titlesize=Big)  
    plt.title(title)
    ax1[0].plot(labels,prooftimes.T,label=labels2,linewidth=2.7)
    ax1[0].legend(labels2)
    ax1[0].set_xlabel('Batch Size')
    ax1[0].xaxis.set_label_position('top') 
    ax1[0].set_ylabel('time (s)')
    ax1[0].xaxis.tick_top()
    stattable=ax1.table(cellText=specs,rowLabels=rowNames,rowColours=colors,colLabels=cLabels,loc= "bottom")
    stattable.auto_set_font_size(False)
    stattable.set_fontsize(12)

    plt.subplots_adjust(left=0.2, bottom=0.3)
    plt.savefig(fname="ProofDiffResEval")

def createPlot_table2(title,labels,labels2,rowNames,prooftimes,proofstats):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    prooftimes = np.asarray(prooftimes)
    cLabels = ("Compile Time(s)","Setup Time(s)","Setup (Mb)","Constrains","Pixel")

    
    specs = np.array(proofstats)
    fig ,ax1 = plt.subplots(nrows=2,figsize=(7,5),gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=1.0, wspace=0.2, hspace=0.1)
    plt.suptitle(title,fontweight="bold")
    ax1[0].plot(labels,prooftimes.T,label=labels2,linewidth=2.7)
    ax1[0].legend(labels2)
    ax1[0].set_xlabel('Batch size') 
    ax1[0].set_ylabel('time (s)')
    ax1[1].axis("off")
    ax1[1].axis("tight")
    table1 = ax1[1].table(fontsize=24,cellLoc="center",cellText=specs,rowLabels=rowNames,rowColours=colors,colLabels=cLabels,loc= "center")
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 1.5)
    fig.tight_layout()

    plt.show()
    #plt.savefig(fname="ProofDiffResEval")
def createPlot_table1(title,labels,labels2,rowNames,prooftimes,proofstats):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    prooftimes = np.asarray(prooftimes)
    cLabels = ("Compile Time(s)","Setup Time(s)","Setup (Mb)","Constrains","Proof (MB)")

    
    specs = np.array(proofstats)
    fig ,ax1 = plt.subplots(nrows=2,figsize=(7,5),gridspec_kw={'height_ratios': [5, 1]})
    plt.subplots_adjust(left=-0.5, right=0.9, bottom=0.1, top=1.0, wspace=0.0, hspace=0.1)
    plt.suptitle(title,fontweight="bold")
    ax1[0].plot(labels,prooftimes.T,label=labels2,linewidth=2.7)
    ax1[0].legend(labels2)
    ax1[0].set_xlabel('Batch Size') 
    ax1[0].set_ylabel('time (s)')
    ax1[0].annotate("257",(3.9,470),fontsize=10,color=colors[4],weight='bold')
    ax1[0].plot(4, 270, "o",color=colors[4])
    ax1[0].annotate("4868",(3.9,5000),fontsize=10,color=colors[0],weight='bold')
    ax1[0].plot(4, 4868, "o",color=colors[0])
    ax1[0].annotate("7743",(3.6,7443),fontsize=10,color=colors[3],weight='bold')
    ax1[0].plot(4, 7743, "o",color=colors[3])
    pos = ax1[0].get_position()
    newpos =[pos.x0+0.4, pos.y0, pos.width-0.4, pos.height]
    ax1[0].set_position(newpos)
    ax1[1].axis("off")
    ax1[1].axis("tight")
    pos1 = ax1[1].get_position()
    ax1[1].set_position([pos.x0,pos1.y0,pos.width,pos1.height])
    table1 = ax1[1].table(fontsize=24,cellLoc="center",cellText=specs,rowLabels=rowNames,rowColours=colors,colLabels=cLabels,loc= "center")
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 1.5)
    fig.tight_layout()

    #plt.show()
    plt.savefig(fname="ProofGenerationTimes")
def createPlot_table1CON(title,labels,labels2,rowNames,prooftimes,proofstats):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    prooftimes = np.asarray(prooftimes)
    cLabels = ("Compile Time(s)","Setup Time(s)","Setup (Mb)","Constrains","Proof (MB)")

    
    specs = np.array(proofstats)
    fig ,ax1 = plt.subplots(nrows=2,figsize=(7,5),gridspec_kw={'height_ratios': [5, 1]})
    plt.subplots_adjust(left=-0.5, right=0.9, bottom=0.1, top=1.0, wspace=0.0, hspace=0.1)
    plt.suptitle(title,fontweight="bold")
    ax1[0].plot(labels,prooftimes.T,label=labels2,linewidth=2.7)
    ax1[0].legend(labels2)
    ax1[0].set_xlabel('Batch Size') 
    ax1[0].set_ylabel('time (s)')
    ax1[0].annotate("266",(3.9,470),fontsize=10,color=colors[4],weight='bold')
    ax1[0].plot(4, 270, "o",color=colors[4])
    ax1[0].annotate("4881",(3.9,5000),fontsize=10,color=colors[0],weight='bold')
    ax1[0].plot(4, 4868, "o",color=colors[0])
    ax1[0].annotate("7758",(3.6,7443),fontsize=10,color=colors[3],weight='bold')
    ax1[0].plot(4, 7743, "o",color=colors[3])
    pos = ax1[0].get_position()
    newpos =[pos.x0+0.4, pos.y0, pos.width-0.4, pos.height]
    ax1[0].set_position(newpos)
    ax1[1].axis("off")
    ax1[1].axis("tight")
    pos1 = ax1[1].get_position()
    ax1[1].set_position([pos.x0,pos1.y0,pos.width,pos1.height])
    table1 = ax1[1].table(fontsize=24,cellLoc="center",cellText=specs,rowLabels=rowNames,rowColours=colors,colLabels=cLabels,loc= "center")
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 1.5)
    fig.tight_layout()

    plt.show()
    #plt.savefig(fname="ProofDiffResEval")
if __name__ =="__main__":
    main()