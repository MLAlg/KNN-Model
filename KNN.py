#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:10:12 2019

@author: samaneh
"""

# learning algorithm - K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
from dataset import load_hoda

X_train, y_train, X_test, y_test = load_hoda()
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# prediction on a sample of data
sample = 5
X = [X_test[sample]]
predicted_class = neigh.predict(X)
actual_class = y_test[sample]
print("sample {} is a {}, and your prediction is {}.".format(sample, actual_class, predicted_class))
print(neigh.predict_proba(X))

# prediction on test set
pred_class = neigh.predict(X_test)
true_class = y_test
print("Predicted classes: \n", pred_class)
print("True classes: \n", true_class)
acc = neigh.score(X_test, y_test)
print("Accuracy is: % .2f %%" %(acc*100))
