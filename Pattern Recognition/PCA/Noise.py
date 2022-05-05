# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 09:29:46 2020

@author: punit
"""


import idx2numpy 
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from PCA_python import  PCA


#data imports
file_train='train-images-idx3-ubyte'
arr=idx2numpy.convert_from_file(file_train)
file_label='train-labels-idx1-ubyte'
label=idx2numpy.convert_from_file(file_label)

digit=[[] for i in range(10)]

#noise addition
for i in range(len(arr)):
    t=label[i]
    row,column=arr[i].shape
    sigma=0.2
   
    new = random_noise(arr[i], var=sigma ** 2)
    flatten=new.reshape(row*column,)
    digit[t].append(flatten)
for j in range(10):
    digit[j]=np.asarray(digit[j])
    print(len(digit[j])) 

for i in range(10):
    file=str(i)+'_noise.npy'
    np.save(file,digit[i])
# plt.imshow(digit[4][0].reshape(28,28),cmap='gray')
# plt.show()

if __name__=="__main__":
    number=int(input("Enter digit: "))
    file=str(number)+'_noise.npy'
    list_image=np.load(file)
    plt.imshow(list_image[0].reshape(28,28), cmap='gray')
    plt.show()
    print('Dimension of list of images = ',list_image.shape)
    lower=int(input("enter the lower value of number of eigen vectors: "))
    upper=int(input("enter the upper value of number of eigen vectors: "))
    recons=PCA(list_image,lower,upper)
    print(recons.shape)
    face1=recons[0].reshape(28,28)
    plt.imshow(np.clip(face1,0,255),cmap='gray');plt.show()