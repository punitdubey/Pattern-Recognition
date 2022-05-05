# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 03:55:16 2020

@author: punit
"""
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import predict
from functions import perceptron,split_data
from functions import plot_decision_boundry
from functions import confusion_matrix,properties
#import data
class1 = pd.read_csv("non_linearly_seperable_data\Class1.txt",sep="\t",header = None).drop([2],axis=1)
class2 = pd.read_csv("non_linearly_seperable_data\Class2.txt",sep="\t",header = None).drop([2],axis=1)

#to numpy
class1 = class1.to_numpy()
class2 = class2.to_numpy()
alldata = np.concatenate((class1,class2), axis=0)
#concatenate
alldata = np.concatenate((np.ones((alldata.shape[0],1),dtype=float),alldata),axis=1)

# final outcome should be
desired_vector = np.ones(alldata.shape[0])
desired_vector[alldata.shape[0]//2:] = -1


# final outcome should be
desired_vector = np.ones(alldata.shape[0])
desired_vector[alldata.shape[0]//2:] = -1
#plot graph for Non linear
plt.figure(figsize=(10,10))
plt.scatter(class1[:,0],class1[:,1], color='green',label = "Class1")
plt.scatter(class2[:,0], class2[:,1], color='red',label = "Class2")
plt.title("Non-linear-data")
plt.legend()
plt.show()

#divide data into 2 parts
train_class1,test_class1 = split_data(class1)
train_class2,test_class2 = split_data(class2)
desire_train1,desire_test1 =split_data(desired_vector[0:1000])
desire_train2,desire_test2 =split_data(desired_vector[1000:])
#train
train = np.concatenate((train_class1,train_class2), axis=0) 
train = np.concatenate((np.ones((train.shape[0],1),dtype=float),train),axis=1) 
desire_train = np.concatenate((desire_train1,desire_train2), axis=0)
#test
test = np.concatenate((test_class1,test_class2), axis=0) 
test = np.concatenate((np.ones((test.shape[0],1),dtype=float),test),axis=1) 
desire_test = np.concatenate((desire_test1,desire_test2), axis=0)
#call perceptron
final_weight,error_iteration = perceptron(train,desire_train,iteration=2000)
classify = predict(test,final_weight)
temp1_1 = []
temp1_2 = []
temp2_1 = []
temp2_2 = []

for i in range(classify.shape[0]):
    if  classify[i]==1:
        if desire_test[i]==1:
            temp1_1.append(test[i])
        else:
            temp1_2.append(test[i])
    elif classify[i]==-1:
        if desire_test[i]==-1:
            temp2_2.append(test[i])
        else:
            temp2_1.append(test[i])
temp1_1 = np.array(temp1_1)
temp1_2 = np.array(temp1_2)
temp2_1 = np.array(temp2_1)
temp2_2 = np.array(temp2_2)



# plot
intercept = -(final_weight[0]/final_weight[2])
theta = -(final_weight[1]/final_weight[2])
y_line = train*theta + intercept
plt.figure(figsize=(8,8))
plt.scatter(class1[:,0],class1[:,1], color='red',label = "Class1")
plt.scatter(class2[:,0], class2[:,1], color='green',label = "Class2")
plt.plot(train[:,1:],y_line[:,1:],color='blue',)
if(len(temp1_1)!=0):
    plt.scatter(temp1_1[:,1],temp1_1[:,2],color="darkred",label="test_class1")
if(len(temp2_2)!=0):
    plt.scatter(temp2_2[:,1],temp2_2[:,2],color='darkgreen',label="test_class2")
if(len(temp1_2)!=0):
    plt.scatter(temp1_2[:,1],temp2_2[:,2],color='black',label="wrong_class1")
if(len(temp2_1)!=0):
    plt.scatter(temp2_1[:,1],temp2_1[:,2],color='black',label="wrong_class2")
plt.title("Non-linear-data")
plt.legend()
plt.show()


#iterattions
plt.figure(figsize=(8,8))
plt.plot(error_iteration[:,0],error_iteration[:,1],color = 'blue')
plt.title("error_itteration")
plt.ylabel("error")
plt.xlabel("iterarion")
plt.show()

plot_decision_boundry(alldata,desired_vector,final_weight)
x = confusion_matrix(len(temp1_1),len(temp1_2),
                  len(temp2_1),len(temp2_2))
precision,recall,accuracy,f_measure = properties(x)
print("for Non-linear-data:")
print("precesion : ",precision)
print("recall: ",recall)
accuracy = ((len(temp1_1)+len(temp2_2))*100)/(len(test_class1)+len(test_class2))
print("accuracy : ",accuracy)
print("f_measur :" ,f_measure)
