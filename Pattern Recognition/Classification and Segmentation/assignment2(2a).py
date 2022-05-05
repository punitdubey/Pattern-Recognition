# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:39:11 2020

@author: punit

Perform k-means clustering-based segmentation of the given image
When using only pixel colour values as features
"""
#imports
import numpy as np
import cv2


#functions
def ecludianDistance(initial,newdis):
    """find the ecludian distance between the two points"""
    x1,y1,z1=initial
    x2,y2,z2=newdis
    x = (x1-x2)**2
    y = (y1-y2)**2
    z = (z1-z2)**2
    return np.sqrt(x+y+z)

def newmean(data):
    """find the mean of the data"""
    return np.mean(data,axis=0)

#image read
image = cv2.imread("image.jpg")
image = cv2.resize(image, (500,500)) 
cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#image to numpy 
img = np.array(image)
shape = img.shape

#change image from 2d from 3color(RGB)
pixel_vals = img.reshape((-1,3)) 

 
# Convert to float type 
pixel_vals = np.float32(pixel_vals)
pixel_vals = np.true_divide(pixel_vals,255)
print(pixel_vals)
# kmeans
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
    centers = centers*255    
    centers = np.uint8(centers)
    temp = np.array(temp)
    res = centers[temp]
    print(centers)
    res2 = res.reshape((shape))
    print(res2)
    string1 = "for  K : "+str(i)
    cv2.imshow(string1,res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

        
          
          