# Input parameters
rootdir = 'E:/Academics/2017 Fall/CSE549 computational biology/Project/project1/train'  # Path of training data
op = open("E:/Academics/CBProject/output_EQ_Max.csv",
          "w")  # Path of the target file to which the processed data is written
op1 = open("E:/Academics/2017 Fall/CSE549 computational biology/Project/project1/p1_train.csv",
           "r")  # Input file with labels for each datapoint

import os
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score

line = op1.readline()
line = op1.readline()  # to ignore first line
labelDict = {}
labelIndexDict = {}
labelIndexDict['CEU'] = 0
labelIndexDict['FIN'] = 1
labelIndexDict['GBR'] = 2
labelIndexDict['TSI'] = 3
labelIndexDict['YRI'] = 4

while (line):
    spline = line.split(',')
    key = spline[0]
    val = spline[1][:-1]
    labelDict[str(key)] = labelIndexDict[str(val)]
    line = op1.readline()

listOfEqClassCounts = []
medianCount=
for root, dirs, files in os.walk(rootdir):
    transcriptsList = []
    for file in files:
        if file.endswith("eq_classes.txt") and root.find("no_bias") == -1:
            # print(os.path.join(root, file))
            file_object = open(os.path.join(root, file), "r")

            lines = file_object.readlines()
            transcriptCount = int(lines[0])
            eqClassesCount = int(lines[1])
            i = 2

            transcriptsECScoreDict = {}
            if (len(transcriptsList) == 0):
                while (i < transcriptCount + 2):
                    line = lines[i].strip('\n').strip('\l')
                    transcriptsList.append(line)
                    transcriptsECScoreDict[line] = 0
                    i += 1
            else:
                i = i + transcriptCount
                j = 0
                while (j < transcriptCount):
                    transcriptsECScoreDict[transcriptsList[j]] = 0
                    j += 1

            # print(line)

            j = 0

            while (j < eqClassesCount):
                line = lines[i + j]
                lineData = line.split('\t')
                countOfTranscriptsInCurrentLine = int(lineData[0])
                countForEC = int(lineData[-1])  # last number in the row
                if(countForEC<medianCount):
                    continue
                listOfEqClassCounts.append(countForEC)
                eqClassScore =0
                k = 1
                while (k < countOfTranscriptsInCurrentLine + 1):
                    indexOfTranscript = int(lineData[k])
                    # transcriptsECScoreDict[transcriptsList[
                    #     indexOfTranscript]] = max(transcriptsECScoreDict[transcriptsList[
                    #     indexOfTranscript]], round(countForEC / countOfTranscriptsInCurrentLine, 4))
                    # Or Only countForEC
                    transcriptsECScoreDict[transcriptsList[
                        indexOfTranscript]] += round(countForEC / countOfTranscriptsInCurrentLine,
                                                     4)  # Or Only countForEC

                    k += 1
                j += 1

                # Write to file:
                for transcript in transcriptsList:
                    op.write(str(transcriptsECScoreDict[transcript]) + ",")
                    # op.write(''.join(tpm[1:]))
                    # folderName = os.path.join(root, file).split("\\")[-3]
                    # ALl files done
                folderName = os.path.join(root, file).split("\\")[-4]
                label = labelDict[folderName]
                op.write(str(label) + '\n')
                print(folderName)

import statistics

medianCount = statistics.median(listOfEqClassCounts)

print (medianCount)
op.close()
