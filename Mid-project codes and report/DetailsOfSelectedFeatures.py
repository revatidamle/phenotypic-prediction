op = open("E:/Academics/ReducedDimensionsDetails.txt","w")
op1 = open("E:/Academics/2017 Fall/CSE549 computational biology/Project/project1/p1_train.csv","r")
rootdir = 'E:/Academics/2017 Fall/CSE549 computational biology/Project/project1/train'


import os
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score
from operator import add


# for subdir, dirs, files in os.walk(rootdir):
#     print ('')
#     "here"
#     for dir in dirs:
#         print (dir)
#
#         print (os.path.join(subdir, file))

line = op1.readline()
line = op1.readline()#to ignore first line



reducedFeatures= [152599, 87536, 52510, 85552, 160708, 30231, 32433, 142790, 44668, 120659, 85486, 20888, 45815, 114896, 197771,
                  163045, 21899, 173400, 94880, 149913, 79185, 24006, 30021, 124545, 117833, 182291, 80138, 191837, 111008, 158722, 137502, 77742, 182909, 185349]
dictOfRNACharacteristics ={}
for root, dirs, files in os.walk(rootdir):
    for file in files:
         if file.endswith(".sf") and not root.endswith("no_bias"):
            print(os.path.join(root, file))
            file_object = open(os.path.join(root, file), "r")

            print (file_object.readline)
            line = file_object.readline()
            line = file_object.readline()#Ignore the column headers
            # characteristics=[]
            index=0
            while (line) :
                characteristics = []
                if index  in reducedFeatures:
                    characteristics.append(float(line.split('\t')[1]))
                    characteristics.append(float(line.split('\t')[2]))
                    characteristics.append(float(line.split('\t')[3]))
                    characteristics.append(float(line.split('\t')[4]))
                    rnaID = line.split('\t')[0]
                    if (rnaID in dictOfRNACharacteristics) == False:
                        dictOfRNACharacteristics[rnaID] = characteristics
                    else:
                        dictOfRNACharacteristics[rnaID] = map(add, dictOfRNACharacteristics[rnaID], characteristics)
                index = index + 1
                line = file_object.readline()

for key, value in dictOfRNACharacteristics.items():
    dictOfRNACharacteristics[key] = [x / 369 for x in dictOfRNACharacteristics[key]] # Assumption: 369 is number of files
    values = ",".join(map(str, dictOfRNACharacteristics[key]))
    op.write(str(key)+',' + str(values)+"\n")

op.close()

