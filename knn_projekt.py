# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:38:53 2022

@author: Szymon Feliński

ENG
Examples below show that kNN returns 100% accuracy when testing on training data and k=1.

POL
Na ponizszych zestawach danych zostalo pokazane, ze przy k=1 kNN zawsze daje 100% trafnosci przy testowaniu na danych uczacych.

"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier # classifier library
from sklearn.metrics import accuracy_score # result scoring library
from sklearn.model_selection import train_test_split # train, test data splitting
from sklearn.preprocessing import LabelEncoder # used for categorical data

def select_dataset(selected_dataset):
    if selected_dataset == 0:
        # WINE DATA
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
        # the same data, but locally
        # url = 'wine.data' 
        
        wine_data = pd.read_csv(url, sep=",", header=None)
        
        wine_data.head()
        
        y = 0 # y is first column, the target class
        
        labels = wine_data[y]
        
        features = wine_data.drop(y, axis=1)
        
        # kNN classifier with k=1 neighbors
        wine_classifier_kNN = KNeighborsClassifier(n_neighbors=1)
        
        # split data into training and testing sets, 80% of data in training set
        wine_xtrain, wine_xtest, wine_ytrain, wine_ytest = train_test_split(features, labels, test_size=0.2, random_state=0)
        
        # train the model
        wine_classifier_kNN.fit(wine_xtrain, wine_ytrain)
        
        wine_pred_train = wine_classifier_kNN.predict(wine_xtrain)
        wine_pred_test = wine_classifier_kNN.predict(wine_xtest)
        
        wine_accuracy_train = accuracy_score(wine_ytrain, wine_pred_train)
        wine_accuracy_test = accuracy_score(wine_ytest, wine_pred_test)
        
        # print accuracy results
        print("Wine train accuracy: ", wine_accuracy_train, "\nWine test accuracy: ", wine_accuracy_test)
    
    elif selected_dataset == 1:
        # CAR DATA
        
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
        # the same data, but locally
        # url = 'car.data' 
        
        car_data = pd.read_csv(url, sep=",", header=None)
        
        y = 6 # y is the last column, the target class
        
        labels = car_data[y]
        
        features = car_data.drop(y, axis=1)
        
        # Encoding categorical features as numbers
        lbl_enc = LabelEncoder()
        
        for i in features.columns:
            features[i] = lbl_enc.fit_transform(features[i])
        
        # check whether the features were properly encoded
        # print(car_data.head())
        
        # kNN classifier with k=1 neighbors
        car_classifier_kNN = KNeighborsClassifier(n_neighbors=1)
        
        # split data into training and testing sets, 80% of data in training set
        car_xtrain, car_xtest, car_ytrain, car_ytest = train_test_split(features, labels, test_size=0.2, random_state=0)
        
        # train the model
        car_classifier_kNN.fit(car_xtrain, car_ytrain)
        
        car_pred_train = car_classifier_kNN.predict(car_xtrain)
        car_pred_test = car_classifier_kNN.predict(car_xtest)
        
        car_accuracy_train = accuracy_score(car_ytrain, car_pred_train)
        car_accuracy_test = accuracy_score(car_ytest, car_pred_test)
        
        # print accuracy results
        print("Car train accuracy: ", car_accuracy_train, "\nCar test accuracy: ", car_accuracy_test)
        
    elif selected_dataset == 2:
        # MUSHROOM DATA
        
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
        # the same data, but locally
        # url = 'agaricus-lepiota.data' 
        
        mushroom_data = pd.read_csv(url, sep=",", header=None)
        
        # print(mushroom_data.head())
        
        y = 0 # y is the first column, the target class    
        labels = mushroom_data[y]
        
        features = mushroom_data.drop(y, axis=1)
        
        # in this example get_dummies will be used to encode categorical features, since there are a lot of columns.
        # Encoding categorical features as numbers
        categories = features.columns
        lbl_enc = LabelEncoder()
        
        for i in categories:
            features[i] = lbl_enc.fit_transform(features[i])
        
        # check whether the features were properly encoded
        # print(features.head())
        
        # kNN classifier with k=1 neighbors
        mushroom_classifier_kNN = KNeighborsClassifier(n_neighbors=1)
        
        # split data into training and testing sets, 80% of data in training set
        mushroom_xtrain, mushroom_xtest, mushroom_ytrain, mushroom_ytest = train_test_split(features, labels, test_size=0.2, random_state=0)
        
        # train the model
        mushroom_classifier_kNN.fit(mushroom_xtrain, mushroom_ytrain)
        
        mushroom_pred_train = mushroom_classifier_kNN.predict(mushroom_xtrain)
        mushroom_pred_test = mushroom_classifier_kNN.predict(mushroom_xtest)
        
        mushroom_accuracy_train = accuracy_score(mushroom_ytrain, mushroom_pred_train)
        mushroom_accuracy_test = accuracy_score(mushroom_ytest, mushroom_pred_test)
        
        # print accuracy results
        print("Mushroom train accuracy: ", mushroom_accuracy_train, "\nMushroom test accuracy: ", mushroom_accuracy_test)
        
    elif selected_dataset == 3:
        # audiology DATA
        
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/audiology/audiology.standardized.data'
        # the same data, but locally
        # url = 'audiology.standardized.data' 
        
        audiology_data = pd.read_csv(url, sep=",", header=None)
        
        # print(audiology_data.head())
        
        y = 70 # y is the last column, the target class    
        labels = audiology_data[y]
        
        features = audiology_data.drop(y, axis=1) #drop class attribute
        features = audiology_data.drop(y-1, axis=1) #drop identifier attribute (unique for every case)
        
        # in this example get_dummies will be used to encode categorical features, since there are a lot of columns.
        # Encoding categorical features as numbers
        categories = features.columns
        lbl_enc = LabelEncoder()
        
        for i in categories:
            features[i] = lbl_enc.fit_transform(features[i])
        
        # check whether the features were properly encoded
        # print(features.head())
        
        # kNN classifier with k=1 neighbors
        audiology_classifier_kNN = KNeighborsClassifier(n_neighbors=1)
        
        # split data into training and testing sets, 80% of data in training set
        audiology_xtrain, audiology_xtest, audiology_ytrain, audiology_ytest = train_test_split(features, labels, test_size=0.2, random_state=0)
        
        # train the model
        audiology_classifier_kNN.fit(audiology_xtrain, audiology_ytrain)
        
        audiology_pred_train = audiology_classifier_kNN.predict(audiology_xtrain)
        audiology_pred_test = audiology_classifier_kNN.predict(audiology_xtest)
        
        audiology_accuracy_train = accuracy_score(audiology_ytrain, audiology_pred_train)
        audiology_accuracy_test = accuracy_score(audiology_ytest, audiology_pred_test)
        
        # print accuracy results
        print("Audiology train accuracy: ", audiology_accuracy_train, "\nAudiology test accuracy: ", audiology_accuracy_test)


# 0 - wine
# 1 - car
# 2 - mushroom
# 3 - audiology
for i in range(4):
    select_dataset(i)
del i