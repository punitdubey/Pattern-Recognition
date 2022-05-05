# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 23:55:24 2020

@author: punit
Perform k-means clustering-based segmentation of the given image,
When using both pixel colour and location values as features

"""
#imports
import numpy as np
import cv2

#functions
def ecludianDistance(initial,newdis):
    """find the ecludian distance between the two points"""
    r1,g1,b1,x1,y1=initial
    r2,g2,b2,x2,y2=newdis
    r = (x1-x2)**2
    g = (x1-x2)**2
    b = (x1-x2)**2
    x = (x1-x2)**2
    y = (y1-y2)**2
    
    
    return int(np.sqrt(r+g+b+x+y))

def newmean(data):
    """find the mean of the data"""
    return np.mean(data,axis=0)

#image read
image = cv2.imread("image.jpg")
image = cv2.resize(image, (500,500)) 
cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#image size
width, height = image.shape[:2]
list1 = []
for i in range(width):
    for j in range(height):
        list1.append(np.array([i,j]))

size_array =  np.array(list1)
print(size_array)

#image to numpy 
img = np.array(image)
shape = img.shape

#change image from 2d from 3color(RGB)
pixel_vals = img.reshape((-1,3)) 
pixel_vals = np.concatenate((pixel_vals,size_array), axis=1)

# Convert to float type 
pixel_vals = np.float32(pixel_vals) 
print(pixel_vals)


#k means
for i in range(1,4):
    indices = np.random.choice(pixel_vals.shape[0],i,replace=False)
    centers = np.array([pixel_vals[_] for _ in indices])
    distance = np.empty((pixel_vals.shape[0],len(centers)))
    # Reapting for convergence
    for _ in range(3):
        clusters  = [[] for _ in centers]
        for k in range(len(centers)):
            index = 0
            for l in pixel_vals:
                distance[index,k]=int(ecludianDistance(centers[k],l))
                index += 1  
        temp = []       
        for n in range(distance.shape[0]):
            km  = np.argmin(distance[n])
            temp.append(km)
            clusters[km].append(pixel_vals[n].tolist())
        
        #finding the new center
        for l in range(len(clusters)):
            clusters[l] = np.array(clusters[l])
            centers[l] = newmean(clusters[l])
        
    centers = np.uint8(centers[::,:3])
    temp = np.array(temp)
    res = centers[temp]
    res2 = res.reshape((shape))
    print(res2)
    string1 = "for  K : "+str(i)
    cv2.imshow(string1,res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

