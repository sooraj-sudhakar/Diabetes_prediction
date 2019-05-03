#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 07:57:20 2019

@author: sooraj
"""
path="<project-folder-path>"

filepath=path+"bulk_predict.csv"

import os
import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from IPython.display import clear_output
from sklearn.model_selection import train_test_split

bulk_test=pd.read_csv(filepath)

# Here the last row of the read data consits of the actual predictions. We will exclude
# this data and then pass the rest of the data to the trained neural network for prediction
# We can then compare the predicted values with the original ones to see how much accuracy
# our model has.

# removing the last outcome column
col_names=list(bulk_test.columns)

# Before doing the prediction we will put all the exiting values into list for
# later comparision

existing_out=[]

for i in range(0,len(bulk_test)):
    existing_out.append(bulk_test['Outcome'][i])

bulk_test=bulk_test.iloc[:, :-1]
bulk_test
predicted_output=[]
accuracy=[]
predicted_label=[]

model = load_model(path+'weights.h5')

for j in range(0,len(bulk_test)):
    in_data=list(bulk_test.iloc[j,:])
    in_data=np.asarray(in_data)
    in_data=in_data.reshape(1,-1)
    in_data=preprocessing.normalize(in_data)
    predicted_out=model.predict_classes(in_data)
    if(predicted_out[0][0]==1):
        print("\nThe patient has diabetes !")
        predicted_label.append("Diabetes")
    else :        
        print("\nThe patient dosen't have diabetes !")
        predicted_label.append("No diabetes")
    print("The predicted outcome is : "+str(predicted_out[0][0]))
    predicted_output.append(predicted_out[0][0])

for k in range(0,len(predicted_output)):
    temp=predicted_output[k]-existing_out[k]
    if(temp!=0):
        accuracy.append(temp)

model_accuracy=((len(predicted_output)-len(accuracy))/len(predicted_output))*100
print('The model accuracy is : '+str(model_accuracy)+" %")

bulk_test=pd.read_csv(filepath)
m = np.array(predicted_output)
n=np.array(predicted_label)
bulk_test['Predicted output']=m
bulk_test['Predicted class']=n
bulk_test.to_csv(path+"Predicted_output.csv")

# Graph plotting portion
import matplotlib.pyplot as plt

plt.style.use('ggplot')
x = ['No diabetes', 'diabetes']

print("\nPeople without diabetes : "+str(predicted_output.count(0)))
print("People with diabetes : "+str(predicted_output.count(1)))
energy = [predicted_output.count(0),predicted_output.count(1)]

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, energy, color='green')
plt.xlabel("Diabetes stats")
plt.ylabel("People count")
plt.title("Diabetes distribution")
plt.xticks(x_pos, x)
plt.savefig('Bulk_prediction_out.png')
plt.show()

