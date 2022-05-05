# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 20:47:02 2020

@author: punit

Assignment 2 
For the dataset of Assignment 1, perform classification using k-means clustering for the
non-linearly separable case

"""

#imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('ggplot')

#variables 
distortion= []


#data 
class1 = pd.read_csv("non_linearly_seperable data\Class1.txt",sep="\t",header = None).drop([2],axis=1)
class2 = pd.read_csv("non_linearly_seperable data\Class2.txt",sep="\t",header = None).drop([2],axis=1)

#to numpy
class1 = class1.to_numpy()
class2 = class2.to_numpy()
alldata = np.concatenate((class1,class2), axis=0)


#plot graph for linear
x = class1[:,0]
y = class1[:,1]
x1 = class2[:,0]
y1 = class2[:,1]
plt.scatter(x, y, color='red')
plt.scatter(x1, y1, color='green')
plt.title("non-linear Data")
plt.show()


#functions
def split_data(dataset):
    """divide data into two parts trainging and testing"""
    
    # Define a size for your train set 
    train_size = int(0.7 * len(dataset))
    
    # Split your dataset 
    train_set = dataset[:train_size]
    test_set= dataset[train_size:]
    return train_set,test_set

def ecludianDistance(initial,newdis):
    """find the ecludian distance between the two points"""
    x1,y1=initial
    x2,y2=newdis
    x = (x1-x2)**2
    y = (y1-y2)**2
    return np.sqrt(x+y)

def newmean(data):
    """find the mean of the data"""
    return np.mean(data,axis=0)

#for elbow method
for i in range(1,7):
    indices = np.random.choice(alldata.shape[0],i,replace=False)
    centers = np.array([alldata[_] for _ in indices])
    distance = np.empty((alldata.shape[0],len(centers)))
    temp_distance = 0
    # Reapting for convergence
    for _ in range(5):
        clusters  = [[] for _ in centers]
        for k in range(len(centers)):
            index = 0
            for l in alldata:
                distance[index,k]=ecludianDistance(centers[k],l)
                index += 1  
        
        #for clusters data
        total_distance = 0
        for n in range(distance.shape[0]):
            km  = np.argmin(distance[n])
            total_distance += distance[n,km]
            clusters[km].append(alldata[n].tolist())
        
        #finding the new center
        for l in range(len(clusters)):
            clusters[l] = np.array(clusters[l])
            centers[l] = newmean(clusters[l]) 
        temp_distance = total_distance
    distortion.append(temp_distance)

#plot for elbow method
k = [i for i in range(1,7)]
plt.plot(k,distortion,color='blue') 
plt.title("Elbow method") 
plt.xlabel("k clusters")
plt.ylabel("cost")
plt.show()     

#divide data into 2 parts
train_class1,test_class1 = split_data(class1)
train_class2,test_class2 = split_data(class2)

#initial values
newMean1 = class1[0]
newMean2 = class1[1]

# for k=2
for i in range(5):
    temp1 = []
    temp2 = []
    l1=l2=l3=l4=0
    for j in train_class1:
        distance1 = ecludianDistance(newMean1,j)
        distance2 = ecludianDistance(newMean2,j)
        if distance1 <= distance2:
            temp1.append(j)
            
        else:
            temp2.append(j)
        
    for k in train_class2:
        distance1 = ecludianDistance(newMean1,k)
        distance2 = ecludianDistance(newMean2,k)
        if distance1 <= distance2:
            temp1.append(k)
        
        else:
            temp2.append(k)
            
    temp1 = np.array(temp1)
    temp2 = np.array(temp2)
   
    # plt.plot([newMean1[0],newMean2[0]],[newMean1[1],newMean2[1]],color='blue')
    plt.scatter(temp1[:,0], temp1[:,1],color='red')
    plt.scatter(temp2[:,0],temp2[:,1],color='green')
    plt.show()
    newMean1 = newmean(temp1)
    newMean2 =newmean(temp2)
   