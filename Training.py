#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 22:15:36 2019

@author: Sooraj
"""
path="<project_folder_path>"

import os
import numpy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

os.chdir(path)
dataset = numpy.loadtxt(path+"prima-indians-diabetes.csv", delimiter=",")
print("The total number of entires are : "+str(len(dataset))+"\n")
print(dataset[0])

# split into input (X) and output (Y) variables, splitting csv data
X = dataset[:,0:8]
Y = dataset[:,8]

# split X, Y into a train and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
x_train=preprocessing.normalize(x_train)
y_train=y_train
x_test=preprocessing.normalize(x_test)
y_test=y_test

print("\n The train data sample : ")
print(x_train)
print("\n Train data shape : ")
print(x_train.shape)
print("\nTrain data labels : ")
print(y_train[0:9])
print("\n The test data sample :")
print(x_test)


# Neural network
#----------------
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from IPython.display import clear_output

# random seed for reproducibility
numpy.random.seed(2)

# create model, add dense layers one by one specifying activation function
model = Sequential()
model.add(Dense(15, input_dim=8, activation='relu')) # input layer requires input_dim param
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(1, activation='sigmoid')) # sigmoid instead of relu for final probability between 0 and 1

# compile the model, adam gradient descent (optimized)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()

# call the function to fit to the data (training the network)
model.fit(x_train, y_train, epochs = 1000, batch_size=20, validation_data=(x_test, y_test),callbacks=[plot_losses])

# save the model
model.save('weights.h5')

# Prediction
#-----------
sample_data=[1,85,66,29,0,26.6,0.351,31]  # Actual output is 0
test=numpy.array(sample_data)
test=test.reshape(1,-1)
test=preprocessing.normalize(test)
predicted_out=model.predict_classes(test)
if(predicted_out[0][0]==1):
    print("\nThe patient has diabetes !")
else :
    print("\nThe patient dosen't have diabetes !")
print("The predicted outcome is : "+str(predicted_out[0][0]))