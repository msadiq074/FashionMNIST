# -*- coding: utf-8 -*-
"""
Created on Fri May 15 02:15:58 2020

@author: msadi
"""
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset_train = pd.read_csv('fashion-mnist_train.csv')
X_train = dataset_train.iloc[:,1:].values
Y_train = dataset_train.iloc[:,0:1].values

#feature scaling
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
Y_train = to_categorical(Y_train)
sc = MinMaxScaler(feature_range=(0,1))
X_train = sc.fit_transform(X_train)

#reshaping data
X_train = np.reshape(X_train,(60000,28,28,1))

#creating CNN
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Convolution2D(filters=32,kernel_size=[3,3],input_shape=(28,28,1)))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(filters=32,kernel_size=[3,3]))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=10,activation='softmax'))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

classifier.fit(x=X_train,y=Y_train,batch_size=32,epochs=60)

dataset_test=pd.read_csv('fashion-mnist_test.csv')
X_test = dataset_test.iloc[:,1:].values
Y_test = dataset_test.iloc[:,0:1].values
X_test = sc.transform(X_test)

X_test = np.reshape(X_test,(10000,28,28,1))
Y_pred = classifier.predict(X_test)
y_pred = [np.argmax(y, axis=None, out=None) for y in Y_pred]
y_pred = np.array(y_pred)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
x = [cm[i,i] for i in range(10)]
print(sum(x)/100)
