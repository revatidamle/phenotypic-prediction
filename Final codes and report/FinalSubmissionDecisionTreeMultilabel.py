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
import pickle


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
    # print("First Match Accuracy: " + str((countFirstMatch) / len(expectedList)))
    # print("Second Match Count: " + str((countSecondMatch) / len(expectedList)))
    return count / len(expectedList)


op = open("E:\Academics\CBProject\output_TPMMultiLabel.csv", "r")
# op = open("E:\Academics\CBProject\datatest.csv", "r")
dataset = np.loadtxt(op, delimiter=",")
print(dataset.shape)
# separate the data from the target attributes



X = dataset[:, 0:-2]  # X is the array of the features for all the rows
Y = dataset[:, -2:]  # Y is the array of Labels for all the 369 rows (which range from 0-4)

# Y = Y.astype(np.int64)
# print(Y)
n_samples = len(X)
trainPercent = 1.0
testPercent = 0.0
size_training = np.math.floor(n_samples * trainPercent)
size_test = np.math.floor(n_samples * testPercent)
decision_tree = DecisionTreeClassifier()
clfDT = decision_tree.fit(X[:size_training], Y[:size_training:])

expected = Y[size_test:]
# print(expected)
expected = expected.astype(np.int64)
predicted = decision_tree.predict(X[size_test:])
predicted = predicted.astype(np.int64)
# print(predicted)
predicted = np.array(predicted)
expected = np.array(expected)  # [[2,3]\n[1,2]]
# print("Accuracy:", findAccuracy(expected, predicted))

tree.export_graphviz(clfDT, out_file='tree.png')

# print(clfDT.tree_.feature)  # array of nodes splitting feature

# from inspect import getmembers
# print ("getmembers")
# print( getmembers( clfDT.tree_ ) )
# print decision_tree.tree_.feature
b = np.array([-2])
c = np.setdiff1d(clfDT.tree_.feature, b)
print(c)

X_decision_tree = dataset[:, c]
print("Size of data after feature reduction using decision tree : " + str(X_decision_tree.shape))

trainPercent = 1.0
testPercent = 0.0
n_samples = len(X_decision_tree)
print(n_samples)
size_training = np.math.floor(n_samples * trainPercent)
size_test = np.math.floor(n_samples * testPercent)
# svmclassifier = svm.SVC()
# clfSVM = svmclassifier.fit(X_decision_tree[:size_training], Y[:size_training:])

expected = Y[size_test:]
# predicted = clfSVM.predict(X_decision_tree[size_test:])
# print("Accuracy using SVM on the reduced feature set " + str(findAccuracy(expected, predicted)))

# CROSS VALIDATION DT
# Y1 = Y[:][0]
# print(Y1)
# scores = cross_val_score(clfDT, X_decision_tree, Y, cv=5)
# print("Decision Tree 5 Fold:")
# print(scores[0], scores[1], scores[2], scores[3], scores[4])
clfDT = DecisionTreeClassifier()
clfDT.fit(X_decision_tree[:size_training], Y[:size_training:])
expected = Y[size_test:]
predicted = clfDT.predict(X_decision_tree[size_test:])
print(expected.shape)
print(predicted.shape)
print("Accuracy using Decision Tree on the reduced feature set " + str(findAccuracy(expected, predicted)))

from sklearn.externals import joblib

# Making dump file


# CROSS VALIDATION SVM
# scores = cross_val_score(clfSVM, X_decision_tree, Y, cv=5)
# print("SVM 5 Fold:")
# print(scores[0], scores[1], scores[2], scores[3], scores[4])

# Random Forest
clfRF = RandomForestClassifier()
clfRF.fit(X_decision_tree[:size_training], Y[:size_training:])
expected = Y[size_test:]
predicted = clfRF.predict(X_decision_tree[size_test:])
print(expected.shape)
print(predicted.shape)
print("Accuracy using RF on the reduced feature set " + str(findAccuracy(expected, predicted)))

# predicted1 = predicted[0::5]
# predicted2 = predicted[1::5]
# predicted3 = predicted[2::5]
# predicted4 = predicted[3::5]
# predicted5 = predicted[4::5]

scores = cross_val_score(clfRF, X_decision_tree, Y, cv=5)

score1 = cross_val_score(clfRF, X_decision_tree[0], Y[0], cv=5)
score2 = cross_val_score(clfRF, X_decision_tree[1], Y[1], cv=5)

print("5 Fold score" + str(scores.mean()))

# [reducedFeatures_SingleLabel, dtSingleLabel, rfSIngleLabel] = joblib.load('SingleLabelMidsem.pkl')
#
# arr = [c, clfDT, clfRF, reducedFeatures_SingleLabel, dtSingleLabel, rfSIngleLabel]
#
#
# joblib.dump(arr, 'MultiSingleLabel.pkl')
# scores = cross_val_score(clfRF, X_decision_tree, Y1, cv=5)
# print("RF 5 Fold:")
# print(scores[0], scores[1], scores[2], scores[3], scores[4])


# NEW Attempt
# using binary relevance
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB

# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())

# train
classifier.fit(X_decision_tree[:size_training], Y[:size_training:])

joblib.dump(classifier, 'BinaryRelevance.pkl')

# classifier = joblib.load('filename.pkl')

# predict
predictions = classifier.predict(X_decision_tree[size_test:])
# print("Predictions:")
# print(predictions)
# print (len(predictions))
# print("Accuracy using Binary Relevance on the reduced feature set " + str(findAccuracy(expected, predictions)))
print("Accuracy using Binary Relevance on the reduced feature set ")
from sklearn.metrics import accuracy_score

expected_new = []
predicted_new = []

# predicted_new.append(predictions[i][0] for i in range(predictions.shape[0]))
# predicted_new.append(predictions[i][1] for i in range(predictions.shape[0]))
i = 0
predArray = predictions.toarray()
for i in range(expected.shape[0]):
    expected_new.append(expected[i][0])
    predicted_new.append(predArray[i][0])

i = 0
for i in range(expected.shape[0]):
    expected_new.append(expected[i][1])
    predicted_new.append(predArray[i][1])

# newVar = predictions[0]
# print (newVar)
# from scipy import sparse
# expected_csc = sparse.csc_matrix(expected)
# print ("Expected")
# print (expected_csc)
# print ("Predictions")
# print (predictions)

print(accuracy_score(expected_new, predicted_new))
print(findAccuracy(expected, predArray))
