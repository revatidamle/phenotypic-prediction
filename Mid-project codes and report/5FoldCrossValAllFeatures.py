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

op = open("E:\Academics\CBProject\output_TPM.csv", "r")

dataset = np.loadtxt(op, delimiter=",")
print("Initial shape of dataset: ")
print(dataset.shape)

X = dataset[:, 0:-1]  # X is the list of the features for all the 369 rows
Y = dataset[:, -1]  # Y is the list of Labels for all the 369 rows (which range from 0-4)

n_samples = len(X)


# # SVM
# trainPercent = 0.7
# testPercent = 0.3
# n_samples = len(X_decision_tree)
# size_training = np.math.floor(n_samples * trainPercent)
# size_test = np.math.floor(n_samples * testPercent)


# CROSS VALIDATION DT
decision_treeOnReduced = DecisionTreeClassifier()

scores = cross_val_score(decision_treeOnReduced, X, Y, cv=5)
scoresF1 = cross_val_score(decision_treeOnReduced, X, Y, cv=5, scoring='f1_macro')
print("Decision Tree 5 Fold Results:")
print("Scores:" +str(scores))
print("Mean:" +str(scores.mean()))
print("F1 Scores:" +str(scoresF1))
print("F1 Mean:" +str(scoresF1.mean()))

# CROSS VALIDATION SVM
for kernel in ('linear', 'poly', 'rbf'):

    svmclassifier = svm.SVC(kernel=kernel)

    # print("Accuracy using SVM on the reduced feature set " + str(accuracy_score(expected, predicted)))

    scores = cross_val_score(svmclassifier, X, Y, cv=5)
    scoresF1 = cross_val_score(svmclassifier, X, Y, cv=5, scoring='f1_macro')
    print("SVM 5 Fold Results for kernel : " + kernel)
    print("Scores:" +str(scores))
    print("Mean:" +str(scores.mean()))
    print("F1 Scores:" +str(scoresF1))
    print("F1 Mean:" +str(scoresF1.mean()))


# CROSS VALIDATION Random Forest
clfRF = RandomForestClassifier()

scores = cross_val_score(clfRF, X, Y, cv=5)
scoresF1 = cross_val_score(clfRF, X, Y, cv=5, scoring='f1_macro')
print("Random Forest 5 Fold Results:")
print("Scores:" +str(scores))
print("Mean:" +str(scores.mean()))
print("F1 Scores:" +str(scoresF1))
print("F1 Mean:" +str(scoresF1.mean()))
