import os
import pandas as pd
import numpy as np
import math
import statistics
import random
random.seed()
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def regression(x, t, lam):
    A = x.dot(x.T)
    b = x.dot(t)
    penalty = np.identity(A.shape[0]).dot(lam)
    w = np.linalg.inv(penalty+A).dot(b)

    return w


train_i = []
train_t = []
for x in range(1,11):
    #Convert data frames to numpy martices and transpose the input matrix
    input = pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A1/regression-dataset/trainInput"+str(x)+".csv", header=None).to_numpy()
    target = pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A1/regression-dataset/trainTarget"+str(x)+".csv", header=None).to_numpy()

    #Add a leading 1 to each vector of the input
    train_i.append(np.insert(input.T,0,np.ones(input.shape[0]),axis=0).T)
    train_t.append(target)


mses = []
scikit_mses = []


for lam in np.arange(0,4.1,0.1):
    print("lambda: "+str(lam))

    mse_sum = 0

    #Data is split ten ways for 10-fold cross validation
    for i in range(10):

        #Divide data into training set and validation set
        v_set = train_i[i].T
        v_targets = train_t[i]
        t_set = np.concatenate(train_i[:i]+train_i[i+1:]).T
        t_targets = np.concatenate(train_t[:i]+train_t[i+1:])
        
        #Find error for current fold
        w = regression(t_set, t_targets, lam)
        h = w.T.dot(v_set)

        mse_sum += np.mean(np.square(h - v_targets))
    
    mses.append(mse_sum/10)

#Determine accuracy of best lambda on test set
optimal_lambda = np.arange(0,4.1,0.1)[mses.index(min(mses))]
training_set = np.concatenate(train_i).T
training_targets = np.concatenate(train_t)
validation_set = pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A1/regression-dataset/testInput.csv", header=None).to_numpy()
validation_set = np.insert(validation_set.T,0,np.ones(validation_set.shape[0]),axis=0)
validation_targets = pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A1/regression-dataset/testTarget.csv", header=None).to_numpy()

print(validation_set)

w = regression(training_set, training_targets, optimal_lambda)
h = w.T.dot(validation_set)
mse = np.mean(np.square(h - validation_targets))

print("MSE: "+str(mse))


#Generate plot
plt.title('10-fold cross validated multivariate ridge regression')
plt.plot(np.arange(0,4.1,0.1), mses, label='lambda from 0-4')
plt.legend()
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.show()