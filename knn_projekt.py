# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:38:53 2022

@author: szfel
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier #classifier library
from sklearn.metrics import fbeta_score, accuracy_score #result scoring library
from sklearn.model_selection import train_test_split #train, test data splitting

#wine data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
#the same data, but locally
#url = 'wine.data' 

data = pd.read_csv(url, sep=",", header=None)

data.head()

y = 0 #y is first column, the target class

labels = data[y]

features = data.drop(y, axis=1)

#kNN classifier with k=1 neighbors
wine_classifier_kNN = KNeighborsClassifier(n_neighbors=1)

#split data into training and testing sets, 80% of data in training set
wine_xtrain, wine_xtest, wine_ytrain, wine_ytest = train_test_split(features, labels, test_size=0.2, random_state=0)

#train the model
wine_classifier_kNN.fit(wine_xtrain, wine_ytrain)

wine_pred_train = wine_classifier_kNN.predict(wine_xtrain)
wine_pred_test = wine_classifier_kNN.predict(wine_xtest)

wine_accuracy_train = accuracy_score(wine_ytrain, wine_pred_train)
wine_accuracy_test = accuracy_score(wine_ytest, wine_pred_test)

#print accuracy results
print("Train accuracy: ", wine_accuracy_train, "\nTest accuracy: ", wine_accuracy_test)