# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:52:11 2020

@author: punit
"""
#imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import  multivariate_normal


#variable
distortion = []


#data 
class1 = pd.read_csv("data_1\Class1.txt",sep=",",header = None)
class2 = pd.read_csv("data_1\Class2.txt",sep=",",header = None)

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

def mean(data):
     x = sum(data[::,0])/len(data)
     y = sum(data[::,1])/len(data)
     return [x,y]
 
def split_data(dataset):
    """divide data into two parts trainging and testing"""
    # Shuffle your dataset 
    shuffle_dataset = dataset#.sample(frac=1)
    
    # Define a size for your train set 
    train_size = int(0.7 * len(shuffle_dataset))
    
    # Split your dataset 
    train_set = shuffle_dataset[:train_size]
    test_set= shuffle_dataset[train_size:]
    return train_set.to_numpy(),test_set.to_numpy()

def covariance_class(mat):
    means = mean(mat)
    covs =[]
    for j in range(len(means)):
        t_covs = []
        for k in range(len(means)):
            sum = 0
            for i in range(len(mat)):
                sum += ((mat[i][j] - means[j]) * (mat[i][k] - means[k]))
            covariance  = sum/ len(mat)
            t_covs.append(covariance)
        covs.append(t_covs)
    return covs

#for elbow method

for i in range(1,10):
    indices = np.random.choice(alldata.shape[0],i,replace=False)
    centers = np.array([alldata[_] for _ in indices])
    distance = np.empty((alldata.shape[0],len(centers)))
    temp_distance = 0
    # Reapting for convergence
    for _ in range(10):
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
k = [i for i in range(1,10)]
plt.plot(k,distortion,color='blue') 
plt.title("Elbow method") 
plt.xlabel("k clusters")
plt.ylabel("cost")
plt.show() 

#for k = 2:
#initial values
newMean1 = class1[0]
newMean2 = class1[1]
var1 = var2 = 0
for i in range(5):
    temp1 = []
    temp2 = []
    for j in class1:
        distance1 = ecludianDistance(newMean1,j)
        distance2 = ecludianDistance(newMean2,j)
        if distance1 <= distance2:
            temp1.append(j)
        else:
            temp2.append(j)
    for k in class2:
        distance1 = ecludianDistance(newMean1,k)
        distance2 = ecludianDistance(newMean2,k)
        if distance1 <= distance2:
            temp1.append(k)
        else:
            temp2.append(k)
    temp1 = np.array(temp1)
    temp2 = np.array(temp2)
    newMean1 = newmean(temp1)
    newMean2 = newmean(temp2) 
var1 = np.var(temp1)
var2 = np.var(temp2)    

    
    
plt.scatter(temp1[:,0], temp1[:,1],color='red')
plt.scatter(temp2[:,0],temp2[:,1],color='green')
plt.title("K = 2")
plt.show()

#for k = 4:
#initial values
newMean4k1 = class1[0]
newMean4k2 = class1[1]
newMean4k3 = class1[2]
newMean4k4 = class1[3]
for i in range(5):
    temp4k1 = []
    temp4k2 = []
    temp4k3 = []
    temp4k4 = []
    for j in alldata:
        distance1 = ecludianDistance(newMean4k1,j)
        distance2 = ecludianDistance(newMean4k2,j)
        distance3 = ecludianDistance(newMean4k3,j)
        distance4 = ecludianDistance(newMean4k4,j)
        if np.argmin([distance1,distance2,distance3,distance4]) == 0:
            temp4k1.append(j)
        elif np.argmin([distance1,distance2,distance3,distance4]) == 1:
           temp4k2.append(j)
        elif np.argmin([distance1,distance2,distance3,distance4]) == 2:
           temp4k3.append(j)
        elif np.argmin([distance1,distance2,distance3,distance4]) == 3:
           temp4k4.append(j)
    temp4k1 = np.array(temp4k1)
    temp4k2 = np.array(temp4k2)
    temp4k3 = np.array(temp4k3)
    temp4k4 = np.array(temp4k4)
    newMean4k1 = newmean(temp4k1)
    newMean4k2 =newmean(temp4k2) 
    newMean4k3 =newmean(temp4k3) 
    newMean4k4 =newmean(temp4k4) 
   
plt.scatter(temp4k1[:,0], temp4k1[:,1],color='red')
plt.scatter(temp4k2[:,0],temp4k2[:,1],color='green')
plt.scatter(temp4k3[:,0],temp4k3[:,1],color='blue')
plt.scatter(temp4k4[:,0],temp4k4[:,1],color='yellow')
plt.title("K = 4")
plt.show()

def multidimegaussianpdf(data,mean,covariance_mat,d=2):
    inverseCovMat = np.linalg.pinv(covariance_mat)
    sq = np.sqrt(np.dat(covariance_mat))
    x = 1/(((2*np.pi)**d/2)*sq)
    y = np.exp(-0.5*np.transpose((data-mean))*inverseCovMat*(data-mean))
    return x*y

#for k = 2

    
