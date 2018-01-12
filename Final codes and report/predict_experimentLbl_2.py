import sys

import os
from sklearn.externals import joblib
import numpy as np
import os
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import numpy as np
import pickle
from sklearn.metrics import accuracy_score



if (len(sys.argv) < 4):
    print(
        "Arguments expected: predict_experimentLbl.py <address to the model dump>  <address to the test samples root>  <address to the test labels file>")
    exit(0)

model_dump_add = sys.argv[1]
test_samples_add = sys.argv[2]
test_label_add = sys.argv[3]

# Preprocess data for the given test data, create new processed CSV


rootdir = sys.argv[2]

op = open("output_TPM_MultiLabel.csv", "w")
op1 = open(test_label_add, "r")
line = op1.readline()
line = op1.readline()  # to ignore first line
labelDict = {}
labelIndexDict = {}
labelIndexDict['CEU'] = 0
labelIndexDict['FIN'] = 1
labelIndexDict['GBR'] = 2
labelIndexDict['TSI'] = 3
labelIndexDict['YRI'] = 4

# Building a dictionary of sample data folder name and label for it
while (line):
    spline = line.split(',')
    key = spline[0]
    val = spline[1]
    val2 = spline[2].strip('\n')

    labelDict[str(key)] = str(labelIndexDict[str(val)]) + "," + str(val2)
    line = op1.readline()


def findAccuracy(expectedList, predictedList):
    count = 0

    countFirstMatch = 0
    countSecondMatch = 0
    for i in range(len(expectedList)):
        if np.array_equal(expectedList[i], predictedList[i]):
            count += 1
            countFirstMatch += 1
            countSecondMatch += 1
        elif np.array_equal(expectedList[i][0], predictedList[i][0]):
            count += 0.5
            countFirstMatch += 1
        elif np.array_equal(expectedList[i][1], predictedList[i][1]):
            count += 0.5
            countSecondMatch += 1

    return count / len(expectedList)

print ("Processing TPM Data from given test data ...")
for root, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith(".sf") and not root.endswith("no_bias"):
            # print(os.path.join(root, file))
            file_object = open(os.path.join(root, file), "r")

            line = file_object.readline()
            tpm = []
            while (line):
                tpm.append(line.split('\t')[3] + ',')
                line = file_object.readline()
            op.write(''.join(tpm[1:]))
            folderName = os.path.join(root, file).split("\\")[-3]
            label = labelDict[folderName]
            op.write(str(label) + '\n')
op.close()
#Prepare csv for equivalence class

op = open("output_EqClass.csv", "w")

print ("Processing Equivalence Classes Data from given test data ...")
for root, dirs, files in os.walk(rootdir):
    transcriptsList = []
    for file in files:
        if file.endswith("eq_classes.txt") and root.find("no_bias") == -1:
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


            j = 0

            while (j < eqClassesCount):
                line = lines[i + j]
                lineData = line.split('\t')
                countOfTranscriptsInCurrentLine = int(lineData[0])
                countForEC = int(lineData[-1])  # last number in the row
                # eqClassScore =0
                k = 1
                while (k < countOfTranscriptsInCurrentLine + 1):
                    indexOfTranscript = int(lineData[k])

                    # Or Only countForEC
                    transcriptsECScoreDict[transcriptsList[
                        indexOfTranscript]] += round(countForEC / countOfTranscriptsInCurrentLine,
                                                     4)  # Or Only countForEC
                    # indexOfTranscript]] += round(countForEC)  # Or Only countForEC
                    k += 1
                j += 1

            # Write to file:
            for transcript in transcriptsList:
                op.write(str(transcriptsECScoreDict[transcript]) + ",")
            folderName = os.path.join(root, file).split("\\")[-4]
            label = labelDict[folderName][0]
            op.write(str(label) + '\n')
            # print(folderName)
op.close()




#SC Sequencing Center
[reducedFeatures_SC, dtSC, rfSC, reducedFeatures_EqClass, DTEqClass, RFEqClass, reducedFeatures_multiLabel, DTMultiLabel, RFMultiLabel,
 reducedFeatures_SingleLabel, dtSingleLabel, rfSingleLabel] = joblib.load(model_dump_add)
op = open("output_TPM_MultiLabel.csv", "r")
dataset = np.loadtxt(op, delimiter=",")
X1 = dataset[:, 0:-2]  # X is the array of the features for all the rows
Y1 = dataset[:, -2]  # Y is the array of Labels for all the 369 rows (which range from 0-4 and 1-7)

X_reducedSingleLabel = dataset[:, reducedFeatures_SingleLabel]

print ("***********************")
print ("Results for Single Label Population Prediction:")

predicted = dtSingleLabel.predict(X_reducedSingleLabel)
print("Accuracy for Single Label: Population, using Decision Tree on the reduced feature set: " + str(accuracy_score(Y1, predicted)))
print("F1 Score for Single Label: Population, using Decision Tree on the reduced feature set: " + str(f1_score(Y1, predicted, average = 'micro')))

predicted = rfSingleLabel.predict(X_reducedSingleLabel)
print("Accuracy for Single label: Population, using Random Forest on the reduced feature set: " + str(accuracy_score(Y1, predicted)))
print("F1 Score for Single label: Population, using Random Forest on the reduced feature set: " + str(f1_score(Y1, predicted, average = 'micro')))

print ("***********************")
X1 = dataset[:, 0:-1]  # X is the array of the features for all the rows
Y1 = dataset[:, -1]  # Y is the array of Labels for all the 369 rows (which range from 0-4)

X_reducedSingleLabel = dataset[:, reducedFeatures_SC]


print ("Results for Single Label Sequencing Center Prediction:")
predicted = dtSC.predict(X_reducedSingleLabel)
print("Accuracy for Single Label: Sequencing Center , using Decision Tree on the reduced feature set: " + str(accuracy_score(Y1, predicted)))
print("F1 Score for Single Label: Sequencing Center , using Decision Tree on the reduced feature set: " + str(f1_score(Y1, predicted, average = 'micro')))

predicted = rfSC.predict(X_reducedSingleLabel)
print("Accuracy for Single label: Sequencing Center , using Random Forest on the reduced feature set: " + str(accuracy_score(Y1, predicted)))
print("F1 Score for Single label: Sequencing Center , using Random Forest on the reduced feature set: " + str(f1_score(Y1, predicted, average = 'micro')))



print ("***********************")

# PUT MIDSEM SINGLE






X = dataset[:, 0:-2]  # X is the array of the features for all the rows
Y = dataset[:, -2:]  # Y is the array of Labels for all the 369 rows (which range from 0-4)

print ("Results for Multi Label Prediction:")
X_reduced = dataset[:, reducedFeatures_multiLabel]

predicted = DTMultiLabel.predict(X_reduced)
print("Accuracy for population and sequencing center using Decision Tree on the reduced feature set " + str(findAccuracy(Y, predicted)))
f1ForPop=f1_score(Y[0],predicted[0], average = 'micro')
f1ForSC=f1_score(Y[1],predicted[1], average = 'micro')
f1Score = (f1ForPop+f1ForSC)/2
print("F1 score for population and sequencing center using Decision Tree on the reduced feature set " + str(f1Score))


predicted = RFMultiLabel.predict(X_reduced)
print("Accuracy for population and sequencing center using Random Forest on the reduced feature set " + str(findAccuracy(Y, predicted)))
f1ForPop=f1_score(Y[0],predicted[0], average = 'micro')
f1ForSC=f1_score(Y[1],predicted[1], average = 'micro')
f1Score = (f1ForPop+f1ForSC)/2
print("F1 score for population and sequencing center using Random Forest on the reduced feature set " + str(f1Score))



print ("***********************")
print ("Results for Single Label: Population Prediction using Equivalence Classes:")
op = open("output_EqClass.csv", "r")
dataset = np.loadtxt(op, delimiter=",")
X1 = dataset[:, 0:-1]  # X is the array of the features for all the rows
Y1 = dataset[:, -1]  # Y is the array of Expected Labels
X_reduced = dataset[:, reducedFeatures_EqClass]

predicted = DTEqClass.predict(X_reduced)
print("Accuracy for population using Decision Tree on Equivalence Classes on the reduced feature set " + str(accuracy_score(Y1, predicted)))
print("F1 score for population using Decision Tree on Equivalence Classes on the reduced feature set " + str(f1_score(Y1, predicted, average = 'micro')))

predicted = RFEqClass.predict(X_reduced)
print("Accuracy for population using Random Forest on Equivalence Classes on the reduced feature set " + str(accuracy_score(Y1, predicted)))
print("F1 score for population using Random Forest on Equivalence Classes on the reduced feature set " + str(f1_score(Y1, predicted, average = 'micro')))

op.close()

