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

data = pd.read_csv(url, sep=",", header=None)

data.head()

y = 0 #y is first column, the target class

labels = data[y]

features = data.drop(y, axis=1)

classifier_kNN = KNeighborsClassifier()

#split data into training and testing sets, 80% of data in training set
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2, random_state=0)

#train the model
classifier_kNN.fit(xtrain, ytrain)

