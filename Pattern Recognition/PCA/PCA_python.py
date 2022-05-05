# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 09:22:29 2020

@author: punit
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA

def PCA (list_image,lower,upper):
    mean=np.zeros(list_image[0].shape)
    mean=mean.astype('float32')
    list_image=list_image.astype('float32')
    for i in range(len(list_image)):
        for j in range(len(list_image[i])):
            mean[j]=mean[j]-list_image[i][j]
    for i in range(len(list_image[0])):
        mean[i]=mean[i]/len(list_image)
    print('mean',mean.shape)
    cov= np.cov(list_image.T)
    print('shape of covariance matrix',cov.shape)
    evalue,evector=LA.eigh(cov)
    sorted_index=np.argsort(evalue)[::-1]
    evector=evector[:,sorted_index] 
    evector=evector[:,lower :upper]
    if(upper-lower>10):
        for i in range(10):
            evect=evector.T
            plt.subplot(2,5,i+1),plt.imshow(evect[i].reshape(28,28),cmap='gray')
        plt.show()

    print(evector.shape)
    Y=np.dot(list_image,evector)
    print(Y.shape)
    reconstruct=np.dot(Y,evector.T)
    return reconstruct


if __name__=="__main__":
    number=int(input("Enter digit to reduce: "))
    file=str(number)+'.npy'
    list_image=np.load(file)
    plt.imshow(list_image[0].reshape(28,28), cmap='gray')
    plt.show()
    print('Dimension of list of images = ',list_image.shape)
    lower=int(input("lower value of number of eigen vectors: "))
    upper=int(input("upper value of number of eigen vectors: "))
    recons=PCA(list_image,lower,upper)
    print(recons.shape)
    face1=recons[0].reshape(28,28)
    plt.imshow(np.clip(face1,0,255),cmap='gray');plt.show()