#Kilby Baron 20773955


import os
import pandas as pd
import numpy as np
import math
import statistics
import random
random.seed()
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import time

start_time = time.clock()

#Import test data and training data
test_data = pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A2/dataset/testData.csv", header=None).to_numpy()
test_labels = pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A2/dataset/testLabels.csv", header=None).to_numpy()
t_data = []
t_labels = []
for x in range(1,11):
    t_data.append(pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A2/dataset/trainData"+str(x)+".csv", header=None))
    t_labels.append(pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A2/dataset/trainLabels"+str(x)+".csv", header=None))

#Concatenate training data and labels
training_data = pd.concat(t_data)
training_labels = pd.concat(t_labels)
N = training_labels.shape[0]

#c1 contains indexes of all rows that belong to class 1 (ie label == 5)
c1 = training_labels.iloc[:,0] == 5
c2 = training_labels.iloc[:,0] == 6

#Get the number of data points in each class
num5s = training_labels[c1].shape[0]
num6s = training_labels.shape[0] - num5s

#Calculate class probability (pi)
pi1 = num5s/N
pi2 = num6s/N

#The mean vectors contain the means of each column that belongs that class
mean1 = training_data[c1].mean(axis=0)
mean2 = training_data[c2].mean(axis=0)

#vn contains all rows belonging to class n, the mean of each column from that class subtracted from each value
v1 = (training_data[c1]-mean1).to_numpy()
v2 = (training_data[c2]-mean2).to_numpy()

#Sn multiplies vn by vn transpose and then divides each value by the number of values in class n
S1 = (np.dot(v1.T,v1)/num5s)
S2 = (np.dot(v2.T,v2)/num5s)

#covariance matrix calculation
covariance_m = np.add((num5s/N)*S1,(num6s/N)*S2)
covariance_inv = np.linalg.inv(covariance_m)

#calculate parameters of sigmoid function
w = np.dot(covariance_inv,np.subtract(mean1,mean2))
w0 = -0.5*np.linalg.multi_dot([mean1.T, covariance_inv, mean1])+0.5*np.linalg.multi_dot([mean2.T,covariance_inv,mean2])+math.log(pi1/pi2)

#Form predictions
correct_predictions = 0
for i in range(0,test_data.shape[0]):
    probability = 1/(1+math.exp(-1*np.dot(w.T,test_data[i])+w0))
    if probability >= 0.5:
        prediction = 5
    else:
        prediction = 6
    if prediction == test_labels[i]:
        correct_predictions += 1

print("MIXTURE OF GAUSSIANS RESULTS")
print(correct_predictions,"/",test_data.shape[0])
print("pi:",pi1)
print("mean1:",np.array(mean1))
print("mean2:",np.array(mean2))
print("covariance matrix diagonal:",covariance_m.diagonal())
print("w:",w)
print("w0:",w0)


time_elapsed = (time.clock() - start_time)

print("Computation time:",time_elapsed)













