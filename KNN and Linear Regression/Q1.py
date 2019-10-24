import os
import pandas as pd
import numpy as np
import math
import statistics
import random
random.seed()
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#Calculate the distance between 2 rows
def distance(new, old):
    squared_sum = 0
    for i in range(len(old)):
        diff = old[i]-new[i]
        squared_sum += diff*diff
    return math.sqrt(squared_sum)

#Return the most common label amongst the k nearest neighbours
def knn(instance, data, labels, k):
    #Find the k nearest neighbours and store them in a list
    #Each element of the list contains a label and a distance
    nns = []
    for r in data.itertuples():
        d = distance(instance,r[1:]) #The first element of r is the row index hence r[1:]
        label = labels.iloc[r.Index,0]
        
        # Add current row as a nearest neighbour if there are less than k neighbours, 
        # or if the current list includes one that is further away
        if len(nns) < k:
            nns.append([label,d])

        #Break ties at random
        elif d == nns[-1][1] and random.randint(0,1) == 1:
            nns[-1][0] = label

        else:
            for i in range(k):
                if nns[i][1] > d:
                    nns.insert(i,[label,d])
                    del nns[-1]
                    break

    #Find the most frequent label of the k nearest neighbours
    fives = 0
    sixes = 0
    for n in nns:
        if n[0] == 5:
            fives += 1
        else:
            sixes += 1

    if fives == sixes:
        return random.randint(5,6) #Break ties at random
    if fives > sixes:
        return 5
    return 6


#Import test data and training data
test_data = pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A1/knn-dataset/testData.csv", header=None)
test_labels = pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A1/knn-dataset/testLabels.csv", header=None)
t_data = []
t_labels = []
for x in range(1,11):
    t_data.append(pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A1/knn-dataset/trainData"+str(x)+".csv", header=None))
    t_labels.append(pd.read_csv("C:/Users/Kilby/Code/Waterloo/CS680/A1/knn-dataset/trainLabels"+str(x)+".csv", header=None))

#This list will be filled with the accuracies of each k
accuracies = []

#Potential k values range from 1 - 30
for k in range(1,31):
    
    k_accuracies = []

    #Data is split ten ways for 10-fold cross validation
    for i in range(10):

        #Divide data into training set and validation set
        v_set = t_data[i]
        v_set_labels = t_labels[i]
        t_set = pd.concat(t_data[:i]+t_data[i+1:])#t_data[:i] WAS WRITTEN AS t_data[:1] -- RERUN WHOLE TEST
        t_set_labels = pd.concat(t_labels[:i]+t_labels[i+1:])

        #Make a label hypothesis for each row of the test set using knn
        #Compare each hypothesis to actual label to determine accuracy
        successes = 0
        for row in range(v_set.shape[0]):
            h = knn(v_set.iloc[row], t_set, t_set_labels, k)
            if h == v_set_labels.iloc[row,0]:
                successes += 1
        k_accuracies.append(successes / v_set.shape[0])

    #The overall accuracy for k is the average accuracy from the 10-fold cross validation
    accuracies.append(statistics.mean(k_accuracies))


#Use the best k value to determine the accuracy on the test set
train_set = pd.concat(t_data)
train_set_labels = pd.concat(t_labels)
successes = 0
for row in range(test_data.shape[0]):
    h = knn(test_data.iloc[row], train_set, train_set_labels, 5)
    if h == test_labels.iloc[row,0]:
        successes += 1

final_accuracy = successes/test_data.shape[0]
print("FINAL ACCURACY: "+str(final_accuracy))


#Plot accuracies of k ranging from 1 to 30
plt.title('KNN 10-fold cross validation')
plt.plot(range(1,31), accuracies, label='k Accuracies')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()
print(accuracies)