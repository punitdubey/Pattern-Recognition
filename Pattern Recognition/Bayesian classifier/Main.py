# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 14:51:38 2020

@author: punit
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#functions
 
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


def mean(data):
     x = sum(data[::,0])/len(data)
     y = sum(data[::,1])/len(data)
     return [x,y]
 
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

def PDF_generate(data, mu, covm,pr=1):
    inverseCovMat = np.linalg.pinv(covm)
    transposeData = np.transpose(data)
    transposemean = np.transpose(mu)
    Upper_w = (-0.5)*(inverseCovMat)
    Lower_w = np.matmul((inverseCovMat),(mu))
    transpose_lw = np.transpose(Lower_w)
    first_term = np.matmul(np.matmul((transposeData),(Upper_w)),(data))
    second_term = np.matmul((transpose_lw),(data))
    third_term = -0.5*(np.matmul(np.matmul(transposemean,inverseCovMat),mu))-0.5*(np.log(np.linalg.det(covm)))
    
    pdf=first_term+second_term+third_term+np.log(pr)
    return pdf

def classifier(data, mean1, cov1, mean2, cov2):    
    class1=[]
    class2=[]
    
    for i in (range(data.shape[0])):
        pdf1 = PDF_generate(data[i],mean1,cov1)
        pdf2 = PDF_generate(data[i],mean2,cov2)
        if(np.argmax([pdf1,pdf2])==0):
            class1.append([data[i][0],data[i][1]])
        elif(np.argmax([pdf1,pdf2])==1):
            class2.append([data[i][0],data[i][1]])
            
    class1 = np.array(class1)
    class2 = np.array(class2)
    return class1, class2 

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


def precesion_recall_accuracy(conmat):
    precision = 0
    recall = 0
    precision = conmat[0][0]/ (conmat[0][0]+conmat[1][0])
    recall = conmat[0][0]/ (conmat[0][0]+conmat[0][1])
    accuracy = (conmat[0][0]+conmat[1][1])/(conmat[0][0]+conmat[0][1]+conmat[1][0]+conmat[1][1])
    f_measure = (2*recall*precision)/(recall+precision)
    return precision,recall,accuracy,f_measure

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

def classifier_three(data, mean1, cov1, mean2, cov2, mean3, cov3):    
    class1=[]
    class2=[]
    class3=[]
    for i in (range(data.shape[0])):
        pdf1 = PDF_generate(data[i],mean1,cov1)
        pdf2 = PDF_generate(data[i],mean2,cov2)
        pdf3 = PDF_generate(data[i],mean3,cov2)
        index = [pdf1,pdf2,pdf3].index((max([pdf1,pdf2,pdf3])))
        if index == 0:
            class1.append([data[i][0],data[i][1],i])
        elif index== 1:
            class2.append([data[i][0],data[i][1],i])
        else:
            class3.append([data[i][0],data[i][1],i])
           
            
    class1 = np.array(class1)
    class2 = np.array(class2)
    class3 = np.array(class3)
    return class1, class2 ,class3

def confusion_matrix_three(len1,len2,len3,len4,len5,
                     len6,len7,len8,len9):
    x  = np.empty([3,3],dtype=int)
    x[0][0] = len1
    x[0][1] = len2
    x[0][2] = len3
    x[1][0] = len4
    x[1][1] = len5
    x[1][2] = len6
    x[2][0] = len7
    x[2][1] = len8
    x[2][2] = len9
    print("Confusion Matrix for real world data:\n")
    for i in range(3):
        for j in range (3):
            print(x[i][j],end="\t\t")
        print("\n")
    return x

#for Linear data
class1 = pd.read_csv("linearly_seperable_data\Class1.txt",sep="\t",header = None).drop([2],axis=1)
class2 = pd.read_csv("linearly_seperable_data\Class2.txt",sep="\t",header = None).drop([2],axis=1)

#plot graph for linear
x = class1[0].to_numpy()
y = class1[1].to_numpy()
x1 = class2[0].to_numpy()
y1 = class2[1].to_numpy()
plt.scatter(x, y, color='red')
plt.scatter(x1, y1, color='green')
plt.title("Linear Data")
plt.show()



#for linear split data
train_class1,test_class1 = split_data(class1)
train_class2,test_class2 = split_data(class2)

#after classification
list1,list2 = classifier(test_class1,
                         mean(train_class1),
                         covariance_class(train_class1),
                         mean(train_class2),
                         covariance_class(train_class2),
                        )
list3,list4 = classifier(test_class2,
                         mean(train_class1),
                         covariance_class(train_class1),
                         mean(train_class2),
                         covariance_class(train_class2),
                        )

x = confusion_matrix(len(list1),len(list2),
                 len(list3),len(list4))
   


#plot

plt.scatter(train_class1[::,0],train_class1[::,1])
plt.scatter(list1[::,0],list1[::,1])
# plt.scatter(list2[::,0],list2[::,1])
plt.scatter(train_class2[::,0],train_class2[::,1])
# plt.scatter(list3[::,0],list3[::,1])
plt.scatter(list4[::,0],list4[::,1])
plt.title("Linear Data Prediction")
plt.show()
precision,recall,accuracy,f_measure = properties(x)
print("for Non-linear: \n\n")
print("precesion : ",precision)
print("recall: ",recall)
accuracy = ((len(list1)+len(list4))*100)/(len(test_class1)+len(test_class2))
print("accuracy : ",accuracy)
print("f_measur :" ,f_measure)     

####non _linear data
class1 = pd.read_csv("non_linearly_seperable data\Class1.txt",sep="\t",header = None).drop([2],axis=1)
class2 = pd.read_csv("non_linearly_seperable data\Class2.txt",sep="\t",header = None).drop([2],axis=1)       
#plot graph
x = class1[0].to_numpy()
y = class1[1].to_numpy()
x1 = class2[0].to_numpy()
y1 = class2[1].to_numpy()
plt.scatter(x, y, color='red')
plt.scatter(x1, y1, color='green')
plt.title("Non Linear Data")
plt.show()        

#split data for non linear      
train_class1,test_class1 = split_data(class1)
train_class2,test_class2 = split_data(class2)
      
#class        
list1,list2 = classifier(test_class1,
                         mean(train_class1),
                         covariance_class(train_class1),
                         mean(train_class2),
                         covariance_class(train_class2),
                        )
list3,list4 = classifier(test_class2,
                         mean(train_class1),
                         covariance_class(train_class1),
                         mean(train_class2),
                         covariance_class(train_class2),
                        )      
x = confusion_matrix(len(list1),len(list2),
                     len(list3),len(list4))       

precision,recall,accuracy,f_measure = properties(x)
print("for Non-linear")
print("precesion : ",precision)
print("recall: ",recall)
accuracy = (len(list1)+len(list4))*100/(len(test_class1)+len(test_class2))
print("accuracy : ",accuracy)
print("f_measur :" ,f_measure)      

# print(metric(x))
plt.scatter(train_class1[::,0],train_class1[::,1],color='red')
plt.scatter(train_class2[::,0],train_class2[::,1],color='green')
plt.scatter(list1[::,0],list1[::,1])
# plt.scatter(list2[::,0],list2[::,1])
# plt.scatter(list3[::,0],list3[::,1])
plt.scatter(list4[::,0],list4[::,1])
plt.title("Linear Data prediction")
plt.show()



#real world data
class1 = pd.read_csv("real_world_data\Class1.txt",sep=" ",header = None).drop([2],axis=1)
class2 = pd.read_csv("real_world_data\Class2.txt",sep=" ",header = None).drop([2],axis=1)
class3 = pd.read_csv("real_world_data\Class3.txt",sep=" ",header = None).drop([2],axis=1)
# plot graph
x = class1[0].to_numpy()
y = class1[1].to_numpy()
x1 = class2[0].to_numpy()
y1 = class2[1].to_numpy()
x2 = class3[0].to_numpy()
y2 = class3[1].to_numpy()
plt.scatter(x, y, color='red')
plt.scatter(x1, y1, color='green')
plt.scatter(x2, y2, color='blue')
plt.title("Real World Data")
plt.show()
# seperating data
train_class1,test_class1 = split_data(class1)
train_class2,test_class2 = split_data(class2)
train_class3,test_class3 = split_data(class3)

#after classification

list1,list2,list3 = classifier_three(test_class1,
                         mean(train_class1),
                         covariance_class(train_class1),
                         mean(train_class2),
                         covariance_class(train_class2),
                         mean(train_class3),
                         covariance_class(train_class3),
                        )
list4,list5,list6 = classifier_three(test_class2,
                         mean(train_class1),
                         covariance_class(train_class1),
                         mean(train_class2),
                         covariance_class(train_class2),
                         mean(train_class3),
                         covariance_class(train_class3),
                        )
list7,list8,list9 = classifier_three(test_class3,
                         mean(train_class1),
                         covariance_class(train_class1),
                         mean(train_class2),
                         covariance_class(train_class2),
                         mean(train_class3),
                         covariance_class(train_class3),
                        )

plt.scatter(train_class1[::,0],train_class1[::,1],color='red')
plt.scatter(train_class2[::,0],train_class2[::,1],color='green')
plt.scatter(train_class3[::,0],train_class3[::,1],color='blue')
plt.scatter(list1[::,0],list1[::,1])
plt.scatter(list2[::,0],list2[::,1])
plt.scatter(list3[::,0],list3[::,1])
# plt.scatter(list4[::,0],list4[::,1])
plt.scatter(list5[::,0],list5[::,1])
plt.scatter(list6[::,0],list6[::,1])
plt.scatter(list7[::,0],list7[::,1])
plt.title("Real World Data prediction")
plt.show()


print("\n\nfor Real World Data")
x = confusion_matrix_three(
                  len(list1),len(list2),
                  len(list3),len(list4),len(list5),len(list6),len(list7),len(list8),
                  len(list9))

precision,recall,accuracy,f_measure = properties(x)
print("precesion : ",precision)
print("recall: ",recall)
accuracy = (len(list1)+len(list5)+len(list9))*100/(len(test_class1)+len(test_class2)+len(test_class3))
print("accuracy : ",accuracy)
print("f_measur :" ,f_measure)     
