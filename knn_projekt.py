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

#0 - wine
#1 - car
#2 - mushroom
#3 - audiology
selected_dataset = 1

if selected_dataset == 0:
    #WINE DATA
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
    #the same data, but locally
    #url = 'wine.data' 
    
    wine_data = pd.read_csv(url, sep=",", header=None)
    
    wine_data.head()
    
    y = 0 #y is first column, the target class
    
    labels = wine_data[y]
    
    features = wine_data.drop(y, axis=1)
    
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
    print("Wine train accuracy: ", wine_accuracy_train, "\nWine test accuracy: ", wine_accuracy_test)

elif selected_dataset == 1:
    #CAR DATA
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
    #the same data, but locally
    #url = 'car.data' 
    
    car_data = pd.read_csv(url, sep=",", header=None)
    
    car_data.head()
    
    y = 6 #y is last column, the target class
    
    labels = car_data[y]
    
    features = car_data.drop(y, axis=1)
    
    #kNN classifier with k=1 neighbors
    car_classifier_kNN = KNeighborsClassifier(n_neighbors=1)
    
    #split data into training and testing sets, 80% of data in training set
    car_xtrain, car_xtest, car_ytrain, car_ytest = train_test_split(features, labels, test_size=0.2, random_state=0)
    
    #train the model
    car_classifier_kNN.fit(car_xtrain, car_ytrain)
    
    car_pred_train = wine_classifier_kNN.predict(car_xtrain)
    car_pred_test = wine_classifier_kNN.predict(car_xtest)
    
    car_accuracy_train = accuracy_score(car_ytrain, car_pred_train)
    car_accuracy_test = accuracy_score(car_ytest, car_pred_test)
    
    #print accuracy results
    print("Car train accuracy: ", car_accuracy_train, "\nCar test accuracy: ", car_accuracy_test)
    
#elif selected_dataset == 2:
    #MUSHROOM DATA