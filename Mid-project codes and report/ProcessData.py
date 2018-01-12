# Input parameters
rootdir = 'E:/Academics/2017 Fall/CSE549 computational biology/Project/project1/train'# Path of training data
op = open("E:/Academics/output_TPM.csv","w") # Path of the target file to which the processed data is written
op1 = open("E:/Academics/2017 Fall/CSE549 computational biology/Project/project1/p1_train.csv","r") # Input file with labels for each datapoint

import os
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score


line = op1.readline()
line = op1.readline()#to ignore first line
labelDict={}
labelIndexDict={}
labelIndexDict['CEU']=0
labelIndexDict['FIN']=1
labelIndexDict['GBR']=2
labelIndexDict['TSI']=3
labelIndexDict['YRI']=4

while (line):
    spline = line.split(',')
    key = spline[0]
    val = spline[1][:-1]
    labelDict[str(key)] = labelIndexDict[str(val)]
    line = op1.readline()

for root, dirs, files in os.walk(rootdir):
    for file in files:
         if file.endswith(".sf") and not root.endswith("no_bias"):
            print(os.path.join(root, file))
            file_object = open(os.path.join(root, file), "r")

            print (file_object.readline)
            line = file_object.readline()
            tpm=[]
            while (line) :
                tpm.append(line.split('\t')[3]+',')
                # op.write(tpm+',')
                # print (tpm)
                line = file_object.readline()
            op.write(''.join(tpm[1:]))
            folderName = os.path.join(root, file).split("\\")[-3]
            label = labelDict[folderName]
            op.write(str(label)+'\n')
op.close()

