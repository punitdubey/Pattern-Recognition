# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 03:50:18 2020

@author: punit
"""

#imports 
import numpy as np
import matplotlib.pyplot as plt

#functions
def predict(dataset,weight):
    classifiy = np.zeros(dataset.shape[0])
    for i in range(dataset.shape[0]):
        val = np.dot(dataset[i],weight)
        if (val>=0) :
            classifiy[i] = 1
        else:
            classifiy[i] = -1
    return classifiy

def perceptron(dataset,desired_vector,iteration=300,learning_rate=0.1):
    """to classify into two class using single layer perceptron"""
    error_iteration = np.zeros((iteration,2))
    weight = np.random.rand(dataset.shape[1])
    for _ in range(iteration):
        error_iteration[_,0] = _
        error = 0
        for i in range(dataset.shape[0]):
            y = np.dot(weight,dataset[i])
            if (y>=0) :
                f = 1
                if f != desired_vector[i]:
                   temp = learning_rate*(desired_vector[i]-f) 
                   weight += temp*dataset[i]
                   error +=1
            else:
                f = -1
                if f != desired_vector[i]:
                   temp = learning_rate*(desired_vector[i]-f) 
                   weight += temp*dataset[i]
                   error +=1
        error_iteration[_,1] =  error
    return weight,error_iteration
            
def split_data(dataset):
    """divide data into two parts trainging and testing"""
    
    # Define a size for your train set 
    train_size = int(0.7 * len(dataset))
    
    # Split your dataset 
    train_set = dataset[:train_size]
    test_set= dataset[train_size:]
    return train_set,test_set

def plot_decision_boundry(dataset,decision,weight):
    h = 0.02
    x_min, x_max = dataset[:,1].min() - 100*h, dataset[:,1].max() + 100*h
    y_min, y_max = dataset[:,2].min() - 100*h, dataset[:,2].max() + 100*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = np.c_[xx.ravel(), yy.ravel()]
    classify = np.zeros(Z.shape[0])
    for i in range(Z.shape[0]):
        val = np.dot(Z[i],weight[1:])
        val += weight[0]
        if (val>=0) :
            classify[i] = 1
        else:
            classify[i] = -1
   
    Z = classify.reshape(xx.shape)
    plt.figure(figsize=(8,8))
    plt.contourf(xx, yy, Z,cmap='RdYlGn',alpha=0.25)
    # add a legend, called a color bar
    plt.contour(xx, yy, Z, colors='black', linewidths=0.5)
    plt.scatter(dataset[:,1], dataset[:,2],c=decision,cmap='RdYlGn')
    plt.title('decision_boundary') 
    plt.show()     

def confusion_matrix(len1,len2,len3,len4):
    x  = np.empty([2,2],dtype=int)
    x[0][0] = len1
    x[0][1] = len2
    x[1][0] = len3
    x[1][1] = len4
    print("Confusion Matrix:\n")
    for i in range(2):
        for j in range (2):
            print(x[i][j],end="\t\t")
        print("\n")
    return x


def properties(conf_matrix):
    x = conf_matrix.shape[0]
    precision = []
    accuracy = []
    recall = []
    f_measure = []
    total_sum = np.sum(conf_matrix)
    for i in range(x):
        TP = conf_matrix[i][i]
        FN = np.sum(conf_matrix[i,:])-conf_matrix[i][i]
        FP = np.sum(conf_matrix[:,i])-conf_matrix[i][i]
        TN = total_sum-(TP+FP+FN)

        recal = TP/(TP+FN)
        preci = TP/(TP+FP)
        acc = (TP+TN)/(TP+TN+FP+FN)
        
        precision.append(preci)
        recall.append(recal)
        accuracy.append(acc)
        f_measure.append((2*recal*preci)/(recal+preci))
        
    return accuracy,precision,recall,f_measure
