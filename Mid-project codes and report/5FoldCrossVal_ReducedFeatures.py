op = open("E:\Academics\CBProject\output_TPM.csv", "r") #Path of CSV file containing the preformatted data.

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


dataset = np.loadtxt(op, delimiter=",")
print("Initial shape of dataset: ")
print(dataset.shape)

X = dataset[:, 0:-1]  # X is the list of the features for all the 369 rows
Y = dataset[:, -1]  # Y is the list of Labels for all the 369 rows (which range from 0-4)

n_samples = len(X)
trainPercent = 1
testPercent = 0
size_training = np.math.floor(n_samples * trainPercent)
size_test = np.math.floor(n_samples * testPercent)
decision_tree = DecisionTreeClassifier()
# decision_tree = DecisionTreeClassifier(criterion="entropy")
clfDT = decision_tree.fit(X[:size_training], Y[:size_training:])


tree.export_graphviz(clfDT, out_file='tree.png')

# (clfDT.tree_.feature)  # gets array of nodes splitting feature

# Removing leaf nodes from the decision tree. The leaves have value=-2
reducedFeatures = list(clfDT.tree_.feature)
reducedFeatures[:] = [item for item in reducedFeatures if item != -2]
print("Indexes of Reduced Features after applying Decision Tree", reducedFeatures)
c = np.array(reducedFeatures)


X_decision_tree = dataset[:, c]
print("Size of data after feature reduction using decision tree : " + str(X_decision_tree.shape))


# CROSS VALIDATION DT
decision_treeOnReduced = DecisionTreeClassifier()
# decision_treeOnReduced = DecisionTreeClassifier(criterion="entropy")
scores = cross_val_score(decision_treeOnReduced, X_decision_tree, Y, cv=5)
scoresF1 = cross_val_score(decision_treeOnReduced, X_decision_tree, Y, cv=5, scoring='f1_macro')
print("Decision Tree 5 Fold Results:")
print("Scores:" +str(scores))
print("Mean:" +str(scores.mean()))
print("F1 Scores:" +str(scoresF1))
print("F1 Mean:" +str(scoresF1.mean()))

# CROSS VALIDATION SVM
for kernel in ('linear', 'poly', 'rbf'):

    svmclassifier = svm.SVC(kernel=kernel)

    scores = cross_val_score(svmclassifier, X_decision_tree, Y, cv=5)
    scoresF1 = cross_val_score(svmclassifier, X_decision_tree, Y, cv=5, scoring='f1_macro')
    print("SVM 5 Fold Results for kernel : " + kernel)
    print("Scores:" +str(scores))
    print("Mean:" +str(scores.mean()))
    print("F1 Scores:" +str(scoresF1))
    print("F1 Mean:" +str(scoresF1.mean()))


# CROSS VALIDATION Random Forest
clfRF = RandomForestClassifier()

scores = cross_val_score(clfRF, X_decision_tree, Y, cv=5)
scoresF1 = cross_val_score(clfRF, X_decision_tree, Y, cv=5, scoring='f1_macro')
print("Random Forest 5 Fold Results:")
print("Scores:" +str(scores))
print("Mean:" +str(scores.mean()))
print("F1 Scores:" +str(scoresF1))
print("F1 Mean:" +str(scoresF1.mean()))
