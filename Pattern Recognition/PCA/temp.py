# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 11:17:01 2020

@author: punit
"""
#imports 
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
#function
def PCA(dataset,n_components):
    """To reduce the dimension of the data"""
    mean = np.mean(dataset,axis=1)
    for i in range(dataset.shape[0]):
        dataset[i] = dataset[i]-mean[i] 
    cov = np.cov(dataset)
    eigen_value,eigen_vector, = la.eigh(cov)
    
    
dataset = np.random.randint(1,100,size =(10,10) )
print(dataset)
PCA(dataset,2)

 # new=arr[i]+level*np.random.normal(10,10,(row,column))