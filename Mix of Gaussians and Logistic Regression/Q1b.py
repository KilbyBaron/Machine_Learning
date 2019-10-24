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

def sigmoid(a):
    return 1/(1+np.exp(-1*a))

def regression(X, C, lam):
    N = X.shape[0]
    M = X.shape[1]

    y = C.copy()
    y[y == 5] = 1
    y[y == 6] = 0

    w = np.random.rand(M)
    for i in range(10):
        #Compute gradient (delta_L) and R
        gradient = np.zeros(M)
        R = np.zeros([N,N])
        for n in range(0,N):
            sig_wtx = sigmoid(np.dot(w.T,X[n]))
            gradient = np.add(gradient,(sig_wtx-y[n])*X[n])
            R[n,n] = sig_wtx*1-sig_wtx
        gradient = np.add(gradient,lam*w)

        #compute Hessian - flipped the order of the transpose due to shape of X
        H = np.linalg.multi_dot([X.T,R,X])
        H = np.add(H,lam*np.identity(M))

        w = w - np.dot(np.linalg.inv(H),gradient)
    
    return w

def predict(test_data,test_labels,w):
    correct_predictions = 0
    for i in range(0,test_data.shape[0]):
        probability = sigmoid(np.dot(w.T,test_data[i]))
        if probability >= 0.5:
            prediction = 5
        else:
            prediction = 6
        if prediction == test_labels[i][0]:
            correct_predictions += 1
    
    return correct_predictions/test_data.shape[0]

#Import test data and training data
test_data = pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A2/dataset/testData.csv", header=None).to_numpy()
test_labels = pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A2/dataset/testLabels.csv", header=None).to_numpy()
t_data = []
t_labels = []
for x in range(1,11):
    t_data.append(pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A2/dataset/trainData"+str(x)+".csv", header=None))
    t_labels.append(pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A2/dataset/trainLabels"+str(x)+".csv", header=None))

#Concatenate training data and labels
training_data = pd.concat(t_data).to_numpy()
training_labels = pd.concat(t_labels).to_numpy()

#Add a 1 to the beginning of each x
training_data = np.insert(training_data.T,0,np.ones(training_data.shape[0]),axis=0).T
test_data = np.insert(test_data.T,0,np.ones(test_data.shape[0]),axis=0).T


#Make cross validation sets first so that I dont remake them for every lambda
cv_sets = []
cv_size = int(training_data.shape[0]/10)
for cv in range(10):

    s = int(cv*cv_size)
    e = int(cv*cv_size+cv_size)
    test = training_data[s:e]
    test_l = training_labels[s:e]
    if cv == 0:
        train = training_data[e:]
        labels = training_labels[e:]
    else:
        train = np.vstack((training_data[0:s],training_data[s:]))
        labels = np.vstack((training_labels[0:s],training_labels[s:]))
    
    y = labels.copy()
    y[y == 5] = 1
    y[y == 6] = 0
    
    cv_sets.append([test,test_l,train,labels,y])


#10-fold cross validation to find optimal lambda
lam_accuracy = []
for lam in np.arange(1500000,2000000,100000):
    print(lam)
    avg_accuracy = 0
    for cv in range(10):

        test = cv_sets[cv][0]
        test_l = cv_sets[cv][1]
        train = cv_sets[cv][2]
        labels = cv_sets[cv][3]

        w = regression(train,labels,lam)
        avg_accuracy += predict(test,test_l,w)

    lam_accuracy.append(avg_accuracy/10)



#Test using test data
w = regression(training_data,training_labels,1805000)
accuracy = predict(test_data,test_labels,w)

print(accuracy)

time_elapsed = time.clock()-start_time
print("Computation time:",time_elapsed)

#Plot accuracies of k ranging from 1 to 30
plt.title('Logistic Regression')
plt.plot(np.arange(1500000,2000000,100000), lam_accuracy, label='accuracy')
plt.legend()
plt.xlabel('lambda')
plt.ylabel('Accuracy')
plt.show()


