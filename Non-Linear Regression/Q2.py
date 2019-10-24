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
from keras.models import Sequential
from keras.layers import Dense
import keras.metrics



#Import test data and training data
test_input = pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A3/nonlinear-regression-dataset/testInput.csv", header=None).to_numpy()
test_target = pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A3/nonlinear-regression-dataset/testTarget.csv", header=None).to_numpy()
#test_input = np.insert(test_input.T,0,np.ones(test_input.shape[0]),axis=0).T #Add row of ones

train_inputs = []
train_targets = []
for x in range(1,11):
    train_input = pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A3/nonlinear-regression-dataset/trainInput"+str(x)+".csv", header=None).to_numpy()
    #train_input = np.insert(train_input.T,0,np.ones(train_input.shape[0]),axis=0).T #Add row of ones
    train_inputs.append(train_input)
    train_targets.append(pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A3/nonlinear-regression-dataset/trainTarget"+str(x)+".csv", header=None).to_numpy())




def plot(range, points, title, x_label):
    plt.title(title)
    plt.plot(range, points, label='MSE')
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel('MSE')
    plt.show()


#Find all basis functions
def x_d(x,d):
    if d == 0:
        return [1]
    new_x = x_d(x,d-1)
    for i in range(d+1):
        new_x.append(np.power(x[0],i)*np.power(x[1],d-i))
    return new_x


#Add dimensions to inputs using basis functions
def add_dimensions(d,data):
    dimensions = int((d*(d+1))/2)   #number of possible monomial basis functions in dimension d
    dX = []
    for i in data:
        dX.append(x_d(i,d))
    return np.array(dX)

#Split training data into cross validation train/test sets
def make_cv_subsets(inputs, targets, x):
     #Cross-validation test subset
    test_in = inputs[x].T
    test_t = targets[x]

    #Cross-validation train subset
    train_in = inputs.copy()
    train_t = targets.copy()
    del train_in[x]
    del train_t[x]
    train_in = np.concatenate(train_in)
    train_t = np.concatenate(train_t)

    return test_in, test_t, train_in, train_t

#Calculate mse when each element is a list of size 1
def mse(pred, targets):
    sum = 0
    for x in range(len(targets)):
        sum += (pred[x]-targets[x])**2
    return sum/len(targets)


def regularized_linear_regression(train_in,train_t,test_in,test_t):
    #Find w using least squares regression
    X = train_in.T  
    A = X.dot(X.T)  #Covaraicen matrix
    b = X.dot(train_t)
    lam = 0.5
    lam_plus_A = lam*np.identity(A.shape[0])+A
    w = np.linalg.inv(lam_plus_A).dot(b)

    #Compute accuracy
    pred = (w.T).dot(test_in).T
    return mse(pred.flatten(),test_t.flatten())


#Regularized generalized linear regression: perform least square regression with the penalty term 0.5wTw
def Qa(inputs, targets, test_input, test_targets):

    mses = dict()
    for d in np.arange(1,5):
        start_time = time.clock()

        #Add dimensions to inputs using basis functions
        d_inputs = []
        for j in range(len(inputs)):
            d_inputs.append(add_dimensions(d,inputs[j]))

        mse_trials = []
        #Perform 10-fold cross validation
        for x in np.arange(len(train_inputs)):

            #Make train/test subsets
            test_in, test_t, train_in, train_t = make_cv_subsets(d_inputs,targets,x)

            #Run regularized linear regression
            mse_trials.append(regularized_linear_regression(train_in,train_t,test_in,test_t))

        mses[d] = statistics.mean(mse_trials)
        print("Run time for degree",d,":",time.clock()-start_time)
    
    print(mses)
    #plot(range(1,5),list(mses.values()),"Regularized Generalized Linear Regression","degree")

    train_input = add_dimensions(4,np.concatenate(inputs))
    train_targets = np.concatenate(targets)
    test_input = add_dimensions(4,test_input)
    test_set_mse = regularized_linear_regression(train_input,train_targets,test_input.T,test_targets)
    print("test set mse:",test_set_mse)





def bayesian_generalized_linear_regression(train_in, train_t, test_in, test_t):
    #Bayesian generalized linear regression:
    phi = train_in.T
    variance = 1
    cov = np.identity(phi.shape[0])
    A_inv = np.linalg.inv(phi.dot(phi.T)+cov)
    phi_y = phi.dot(train_t)
    w_bar = A_inv.dot(phi_y)

    #Prediction
    pred = []
    for i in range(len(test_in.T)):
        x_star = test_in.T[i]
        pred.append(((x_star.T).dot(w_bar)).item()) #The prediction is equal to the mean of the gaussian noise distribution

    return mse(pred,test_t.flatten())


#Bayesian generalized linear regression - assume output noise is gaussian with a variance of 1 and covariance matrix is an identity
def Qb(inputs, targets, test_input, test_targets):

    mses = dict()
    for d in np.arange(1,5):

        start_time = time.clock()

        #Add dimensions to inputs using basis functions
        d_inputs = []
        for j in range(len(inputs)):
            d_inputs.append(add_dimensions(d,inputs[j]))

        mse_trials = []
        #Perform 10-fold cross validation
        for x in np.arange(len(train_inputs)):

            #Make train/test subsets
            test_in, test_t, train_in, train_t = make_cv_subsets(d_inputs,targets,x)

            #Bayesian generalized linear regression:
            mse_trials.append(bayesian_generalized_linear_regression(train_in, train_t,test_in,test_t))

        mses[d] = statistics.mean(mse_trials)

        print("Run time for degree",d,":",time.clock()-start_time)
    
    print(mses)
    #plot(range(1,5),list(mses.values()),"Bayesian Generalized Linear Regression","degree")

    train_input = add_dimensions(4,np.concatenate(inputs))
    train_targets = np.concatenate(targets)
    test_input = add_dimensions(4,test_input)
    test_set_mse = bayesian_generalized_linear_regression(train_input,train_targets,test_input.T,test_targets)
    print("test set mse:",test_set_mse)



def gram(X,k,v=0):
    K = []
    for x1 in X:
        row = []
        for x2 in X:
            row.append(k(x1,x2,v))
        K.append(row)
    
    return np.matrix(K)


#kernel functions dxefined in Q2c
def k1(x1,x2,v=0):
    return (x1).dot(x2)

def k2(x1,x2,v):
    if v != 0:
        x1_sub_x2 = (x1-x2.T).T
        diff = (x1_sub_x2.T).dot(x1_sub_x2)
        return np.exp(-1*diff/(2*v**2))

def k3(x1,x2,d):
    return (k1(x1,x2)+1)**d
        

def gaussian_process_regression(test_inputs, test_targets, X, K, y, kernel_var, k):
    
    pred = []
    for x_star in test_inputs:
        K_plus_I_inv = np.linalg.inv(K+(np.identity(len(K))))
        k_xX = []
        for x in X.T:
            k_xX.append(k(x_star,x,kernel_var))
        k_xX = np.array(k_xX)
        y_star = k_xX.dot(K_plus_I_inv.dot(y))
        pred.append([y_star.item()])
    return pred


#Gaussian process regression
def Qc(train_inputs, train_targets, test_inputs, test_targets):


    #Identity kernel - find mean squared error of the test set
    #--------------------------------------------

    start_time = time.clock()

    X = np.concatenate(train_inputs).T
    y = np.concatenate(train_targets)
    K = gram(X.T,k1)
    pred = gaussian_process_regression(test_inputs, test_targets, X, K, y, None, k1)
    se = mse(pred,test_targets)

    print("Squared Error of Identity kernel:",se) 
    print("Identity kernel run time:",time.clock()-start_time)


    #--------------------------------------------




    #Gaussian kernel - find mse of with variance varying from 1-6 using 10-fold cross validation + mse of test set for best variance
    #----------------------------------------------------------
    mses = dict()
    for v in range(1,7):
        mse_trials = []
        #Perform 10-fold cross validation
        for x in np.arange(len(train_inputs)):
            #Make train/test subsets
            test_in, test_t, X, y = make_cv_subsets(train_inputs,train_targets,x)
            X = X.T
            K = gram(X.T,k2,v)
            pred = gaussian_process_regression(test_in.T, test_t, X, K, y, v, k2)
            mse_trials.append(mse(pred,test_t).flat[0])
        mses[v] = statistics.mean(mse_trials)
    
    print("Cross validation of gaussian kernel:",mses)


    start_time = time.clock()
    X = np.concatenate(train_inputs).T
    y = np.concatenate(train_targets)
    K = gram(X.T,k2,4)
    pred = gaussian_process_regression(test_inputs, test_targets, X, K, y, 4, k2)
    se = mse(pred,test_targets)

    print("Squared Error of Gaussian kernel:",se) 
    print("Run time for Gaussian kernel:",time.clock()-start_time)

    #plot(range(1,7),list(mses.values()),"Gaussian Process: Gaussian Kernel","variance")


                
    #----------------------------------------------------------




    #Polynomial kernel - mse with degrees from 1-4 using 10-fold cross validation + mse of test set for best polynomial degree
    #----------------------------------------------------------

                
    mses = dict()
    for d in range(1,5):
        mse_trials = []
        #Perform 10-fold cross validation
        for x in np.arange(len(train_inputs)):
            #Make train/test subsets
            test_in, test_t, X, y = make_cv_subsets(train_inputs,train_targets,x)
            X = X.T
            K = gram(X.T,k3,d)
            pred = gaussian_process_regression(test_in.T, test_t, X, K, y, d, k3)
            mse_trials.append(mse(pred,test_t).flat[0])
        mses[d] = statistics.mean(mse_trials)
    
    print("Cross validation of polynomial kernel:",mses)


    start_time = time.clock()
    X = np.concatenate(train_inputs).T
    y = np.concatenate(train_targets)
    K = gram(X.T,k3,4)
    pred = gaussian_process_regression(test_inputs, test_targets, X, K, y, 4, k3)
    se = mse(pred,test_targets)

    print("Squared Error of polynomial kernel:",se) 
    print("Run time for polynomial kernel:",time.clock()-start_time)

    #plot(range(1,5),list(mses.values()),"Gaussian Process: Polynomial Kernel","Degree")




    #----------------------------------------------------------


def keras_nn(X,y,test_in,test_t, n_units, nepochs):
     #Prepare model
    model = Sequential()
    model.add(Dense(n_units, input_dim=2,activation='sigmoid'))
    model.add(Dense(n_units, activation='sigmoid'))
    model.add(Dense(1,activation='linear'))
    model.compile(loss="mean_squared_error", optimizer='adam')

    #Train model
    model.fit(X,y,epochs=nepochs, verbose=0)

    #Predict
    pred = model.predict(test_in.T).flatten()

    sum = 0
    test_t = test_t.flatten()
    for i in range(len(pred)):
        sum += (pred[i]-test_t[i])**2
    return sum/len(pred)



#Reference material: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
def Qd(train_input, train_targets, test_input, test_targets):


    nepochs = 4000



    #Running on test set with best number of hidden units: (10)
    se = keras_nn(np.concatenate(train_input),np.concatenate(train_targets),test_input.T,test_targets, 10, nepochs)
    print("Squared error of neural network with 10 hidden units:",se)


    mses = dict()

    for n_units in np.arange(1,11,1):
        start_time = time.clock()

        print(str(n_units)+"-"+str(nepochs))


        mse_trials = []
        #Perform 10-fold cross validation
        for x in np.arange(len(train_inputs)):
            #Make train/test subsets
            test_in, test_t, X, y = make_cv_subsets(train_inputs,train_targets,x)
            mse_trials.append(keras_nn(X,y,test_in,test_t, n_units, nepochs))
        

        mse = statistics.mean(mse_trials)
        str(n_units)+"-"+str(nepochs)
        mses[str(n_units)+"-"+str(nepochs)] = mse


        print("Run time for",n_units,"units:",time.clock()-start_time)


    
    print(mses)
    #plot(range(1,11),list(mses.values()),"Bayesian Generalized Linear Regression","degree")








