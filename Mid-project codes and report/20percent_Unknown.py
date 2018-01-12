op = open("E:\Academics\CBProject\output_TPM.csv", "r")

import numpy as np
import os
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import numpy as np
import math
from sklearn.metrics import confusion_matrix


dataset = np.loadtxt(op, delimiter=",")

reducedDataRowCount = math.floor(dataset.shape[0]*.8)
X = dataset[:reducedDataRowCount, 0:-1]  # X is the list of the features for 80% of total rows
Y = dataset[:reducedDataRowCount, -1]  # Y is the list of Labels for 80% of total rows (which range from 0-4)
print("Number of rows:")
print(len(X))

n_samples = len(X)

decision_tree = DecisionTreeClassifier()
clfDT = decision_tree.fit(X[:n_samples], Y[:n_samples:])

tree.export_graphviz(clfDT, out_file='tree.png')

# (clfDT.tree_.feature)  # gets array of nodes splitting feature

# Removing leaf nodes from the decision tree. The leaves have value=-2
reducedFeatures = list(clfDT.tree_.feature)
reducedFeatures[:] = [item for item in reducedFeatures if item != -2]
print("Index of Reduced Features after applying Decision Tree", reducedFeatures)
c = np.array(reducedFeatures)


X_decision_tree = dataset[:reducedDataRowCount:, c]
print("Size of data after feature reduction using decision tree : " + str(X_decision_tree.shape))


# CROSS VALIDATION DT
decision_treeOnReduced = DecisionTreeClassifier()
clfDTOnReduced = decision_tree.fit(X_decision_tree, Y)

scores = cross_val_score(clfDTOnReduced, X_decision_tree, Y, cv=5)
print("Decision Tree 5 Fold Results:")
print("Scores:" +str(scores))
print("Mean:" +str(scores.mean()))

# CROSS VALIDATION SVM
svmclassifier = svm.SVC(kernel='linear')
clfSVM = svmclassifier.fit(X_decision_tree, Y)


scores = cross_val_score(clfSVM, X_decision_tree, Y, cv=5)
print("SVM 5 Fold Results:")
print("Scores:" +str(scores))
print("Mean:" +str(scores.mean()))

# CROSS VALIDATION Random Forest
clfRF = RandomForestClassifier()
clfRF.fit(X_decision_tree, Y)

scores = cross_val_score(clfRF, X_decision_tree, Y, cv=5)
print("Random Forest 5 Fold Results:")
print("Scores:" +str(scores))
print("Mean:" +str(scores.mean()))

# Predicting over the remaining 20% data
#Decision TREE:
print ("Predicting over the remaining 20% data")
testX = dataset[reducedDataRowCount:, 0:-1]
testY = dataset[reducedDataRowCount:, -1]
textXAfterDimReduction = dataset[reducedDataRowCount:, c]
expected = testY
predicted = clfDT.predict(textXAfterDimReduction)
print("Accuracy using Decision Tree on the Test Data: " + str(accuracy_score(expected, predicted)))
print("Confusion Matrix for Decision Tree on the Test Data:")
cnf_matrix = confusion_matrix(expected, predicted)
print (cnf_matrix)

#SVM
testY = dataset[reducedDataRowCount:, -1]
textXAfterDimReduction = dataset[reducedDataRowCount:, c]
expected = testY
predicted = clfSVM.predict(textXAfterDimReduction)
print("Accuracy using SVM on the Test Data: " + str(accuracy_score(expected, predicted)))

print("Confusion Matrix for SVM on the Test Data:")
cnf_matrix = confusion_matrix(expected, predicted)
print (cnf_matrix)

#Random Forest
testY = dataset[reducedDataRowCount:, -1]
textXAfterDimReduction = dataset[reducedDataRowCount:, c]
expected = testY
predicted = clfRF.predict(textXAfterDimReduction)
print("Accuracy using Random Forest on the Test Data: " + str(accuracy_score(expected, predicted)))

print("Confusion Matrix for Random Forest on the Test Data:")
cnf_matrix = confusion_matrix(expected, predicted)
print (cnf_matrix)