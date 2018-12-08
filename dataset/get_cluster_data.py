# -*- coding: cp936 -*-
import numpy as np
import matplotlib.pyplot as plt
import random

def get_data(dim = 2,classes = 2,count = 1000,train_ratio = 0.8,scale = None,tightness = None,centroids = None,if_show = False):
    '''
    dim -> the dimension of the data vector
    classes -> number of clusters
    count -> total samples
    train_ratio -> train data portion
    scale -> magnitude of the data ,should be > 1
    tightness -> how close the data is to its centroid
    centroids -> array of centers of each cluster, shape should be (B,...), where B is the number of classes
    '''
    if scale is None:
        scale = classes / 2
    elif scale < 1:
        scale = 1
    if tightness is None:
        tightness = 0.05 * scale / classes
        
    if centroids is None:# generate centroids for each class
        centroids = (np.random.rand(classes,dim)  - 0.5) * 2 * scale
        
    X = []
    Y = []

    for i in range(classes): #generate data in each class
        X.append(np.random.normal(0,tightness,(count / classes,dim)) + centroids[i])
        Y += [i] * (count / classes)

    for i in range(count - len(Y)):#pad to required count if division left a remainder
        c_idx = np.random.randint(classes)
        X.append(np.random.normal(0,tightness,(1,dim))+ centroids[c_idx])
        Y.append(c_idx)

    X = np.concatenate(X,0)
    Y = np.array(Y)
    
    p = np.random.permutation(count)
    X = X[p]
    Y = Y[p]
    
    train_count = int(count * train_ratio)
    X_train = X[:train_count]
    X_test = X[train_count:]
    Y_train = Y[:train_count]
    Y_test = Y[train_count:]

    if if_show: # show only the first two dimensions, may use t-sne later
        if dim < 2:
            plt.subplot(121)
            plt.scatter(X_train[:],[0] * len(X_train))
            for i in range(min(classes * 10,int(count * train_ratio))):
                plt.text(X_train[i][0],0,str(Y_train[i]))
            plt.subplot(122)
            plt.scatter(X_test[:],[0] * len(X_test))
            for i in range(min(classes * 10,int(count * (1 - train_ratio)))):
                plt.text(X_test[i][0],0,str(Y_test[i]))
            
        else:
            plt.subplot(121)
            plt.xlim(-1.5 * scale,1.5 * scale)
            plt.ylim(-1.5 * scale,1.5 * scale)
            ax = plt.gca() 
            ax.spines['right'].set_color('none') 
            ax.spines['top'].set_color('none')  
            ax.xaxis.set_ticks_position('bottom')
            ax.spines['bottom'].set_position(('data', 0))
            ax.yaxis.set_ticks_position('left')
            ax.spines['left'].set_position(('data', 0))

            plt.scatter(X_train[:][:,0],X_train[:][:,1])
            for i in range(min(classes * 10,int(count * train_ratio))):
                plt.text(X_train[i][0],X_train[i][1],str(Y_train[i]))
                
            plt.subplot(122)
            plt.xlim(-1.5 * scale,1.5 * scale)
            plt.ylim(-1.5 * scale,1.5 * scale)
            ax = plt.gca() 
            ax.spines['right'].set_color('none') 
            ax.spines['top'].set_color('none')  
            ax.xaxis.set_ticks_position('bottom')
            ax.spines['bottom'].set_position(('data', 0))
            ax.yaxis.set_ticks_position('left')
            ax.spines['left'].set_position(('data', 0))

            plt.scatter(X_test[:][:,0],X_test[:][:,1])
            for i in range(min(classes * 10,int(count * (1 - train_ratio)))):
                plt.text(X_test[i][0],X_test[i][1],str(Y_test[i]))
            
        plt.show()

    return X_train,Y_train,X_test,Y_test
