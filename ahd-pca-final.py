# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 17:12:09 2022

@author: Anders & Bence
"""

## (NOTE) A portion of the code has been copied from the exercises

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
from scipy.linalg import svd

filename = r'C:\Users\Anders\Desktop\5th semester\02450 Intro to Machine Learning and Data Mining\Project 1\African heart disease\ahd.csv'
heart_data = pd.read_csv(filename)
## Cleaning up data
# We notice that there is a binary attribute "famhist" that describes whether 
# a patient has a family history of heart disease or not. We will convert the
# string values "Absent" and "Present" to "0" and "1"
heart_data.famhist = heart_data.famhist.str.replace('Absent', '0')
heart_data.famhist = heart_data.famhist.str.replace('Present', '1')


attributes = ["sbp","tobacco","ldl","adiposity","famhist","typea","obersity","alcohol","age"]


## X,y-format
# If the modelling problem of interest was a classification problem where
# we wanted to classify the chd response attribute, we could now identify obtain
# the data in the X,y-format as so:
data = np.array(heart_data.values, dtype=np.float64)

# Extracing the X matrix from the data set (every column exlcuding chd response)
X = data[:, 1:10]

# chd response attrbiute will be the y-vector
y = data[:,10]


N = X.shape[0]



# PCA
# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components Porject 1');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


#########################################################
# Zero-mean deviation
#########################################################

from scipy.linalg import svd

# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(0)

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T    

# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1
# Plot PCA of the data
f = figure()
title('Afriican Heart Disease Data: PCA')


chd_legend = ["Negative CHD response", "Positive CHD response"]
C = len(chd_legend)     
for c in range(C):

    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(chd_legend)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

show()

###################################

import matplotlib.pyplot as plt
from scipy.linalg import svd
    
r = np.arange(1,X.shape[1]+1)
plt.bar(r, np.std(X,0))
plt.xticks(r, attributes)
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('NanoNose: attribute standard deviations')

## Investigate how standardization affects PCA

###
#Zero-mean AND divide by standard deviation
###

# Subtract the mean from the data
Y1 = X - np.ones((N, 1))*X.mean(0)

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y2 = X - np.ones((N, 1))*X.mean(0)
Y2 = Y2*(1/np.std(Y2,0))
# Here were utilizing the broadcasting of a row vector to fit the dimensions 
# of Y2

# Store the two in a cell, so we can just loop over them:
Ys = [Y1, Y2]
titles = ['Zero-mean', 'Zero-mean and unit variance']
threshold = 0.9
# Choose two PCs to plot (the projection)
i = 0
j = 1

# Make the plot
plt.figure(figsize=(10,15))
plt.subplots_adjust(hspace=.4)
plt.title('African Heart Disease: Effect of standardization')
nrows=3
ncols=2



for k in range(2):
    # Obtain the PCA solution by calculate the SVD of either Y1 or Y2
    U,S,Vh = svd(Ys[k],full_matrices=False)
    V=Vh.T # For the direction of V to fit the convention in the course we transpose
    # For visualization purposes, we flip the directionality of the
    # principal directions such that the directions match for Y1 and Y2.
    if k==1: V = -V; U = -U; 
    
    # Compute variance explained
    rho = (S*S) / (S*S).sum() 
    
    # Compute the projection onto the principal components
    Z = U*S;
    
    
    # Plot projection
    plt.subplot(nrows, ncols, 1+k)
    C = len(chd_legend)
    for c in range(C):
        plt.plot(Z[y==c,i], Z[y==c,j], '.', alpha=.5)
    plt.xlabel('PC'+str(i+1))
    plt.xlabel('PC'+str(j+1))
    plt.title(titles[k] + '\n' + 'Projection' )
    plt.legend(chd_legend)
    plt.axis('equal')
    
    # Plot attribute coefficients in principal component space
    plt.subplot(nrows, ncols,  3+k)
    for att in range(V.shape[1]):
        plt.arrow(0,0, V[att,i], V[att,j])
        plt.text(V[att,i], V[att,j], attributes[att])
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xlabel('PC'+str(i+1))
    plt.ylabel('PC'+str(j+1))
    plt.grid()
    # Add a unit circle
    plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
         np.sin(np.arange(0, 2*np.pi, 0.01)));
    plt.title(titles[k] +'\n'+'Attribute coefficients')
    plt.axis('equal')
            
    # Plot cumulative variance explained
    plt.subplot(nrows, ncols,  5+k);
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.title(titles[k]+'\n'+'Variance explained')

plt.show()


####################################################

# 3D PCA plot

#PCA for zero-mean projection
U,S,Vh = svd(Y1,full_matrices=False)
V=Vh.T
if k==1: V = -V; U = -U; 

# Compute variance explained
rho = (S*S) / (S*S).sum() 

# Compute the projection onto the principal components
Z1 = U*S;


#PCA for zero-mean AND dividng by standard deviation

U,S,Vh = svd(Y2,full_matrices=False)
V=Vh.T
if k==1: V = -V; U = -U; 

# Compute variance explained
rho = (S*S) / (S*S).sum() 

# Compute the projection onto the principal components
Z2 = U*S;

    
  
##
for c in range(C):

    # select indices belonging to class c:
    class_mask = y==c

    plot(Z1[class_mask,i], Z1[class_mask,j], 'o', alpha=.5)
legend(chd_legend)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))
plt.show()
##


import numpy as np
import matplotlib.pyplot as plt

# loop that createsa color color list depending on chd response of patient
color_list = []
for i in y:
    if i == 0:
        color_list.append('b')
    else:
        color_list.append('orange')
        
## Plotting attributes against each to check for correlation

## Adiposity vs. obesity
plt.scatter(X[:,3], X[:,6], c = color_list, alpha=0.5)
plt.title('Adiposity vs. Obesity')
plt.xlabel('Adiposity')
plt.ylabel('Obesity')
plt.show()

## Sbp vs. Age
plt.scatter(X[:,8], X[:,0], c = color_list, alpha=0.5)
plt.title('Age vs. SBP')
plt.xlabel('Age')
plt.ylabel('SBP')
plt.show()

## LDL vs. Tobacco
plt.scatter(X[:,2], X[:,1], c = color_list, alpha=0.5)
plt.title('LDL vs. Tobacco')
plt.xlabel('LDL')
plt.ylabel('Tobacco')
plt.show()


# 3D scatteplot of princiapl components

Xax = Z2[:,0]
Yax = Z2[:,1]
Zax = Z2[:,2]

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111, projection = '3d')

ax.scatter3D(Xax, Yax, Zax, c = color_list)
plt.title("African Heart Disease PCA")
ax.set_xlabel("1st Principal Component")
ax.set_ylabel("2nd Principal Component")
ax.set_zlabel("3rd Principal Component")


# View for zero mean 3d pca
#ax.view_init(25,-145)

# View for zero mean unit variance 3d pca
ax.view_init(15,-55)
# show plot



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
