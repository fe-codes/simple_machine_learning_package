import numpy as np
from sklearn.datasets import fetch_mldata
import random
import matplotlib.pyplot as plt

def get_data(vector_form = False,if_show = False):
    '''
    By default returns X_train.reshape(60000,1,28,28),Y_train,X_test.reshape(10000,1,28,28),Y_test
    if vector_form = True is given, returns X_train,Y_train,X_test,Y_test
    '''
    mnist = fetch_mldata('MNIST original')
    np.random.seed(1234) # set seed for deterministic ordering
    p = np.random.permutation(mnist.data.shape[0])
    X = mnist.data[p]
    Y = mnist.target[p]

    X = X.astype(np.float32)/255
    X_train = X[:60000]
    X_test = X[60000:]
    Y_train = Y[:60000]
    Y_test = Y[60000:]

    if if_show:
        plt.figure()
        for i in range(100):
            plt.subplot(10,10,i + 1)
            plt.xticks([])
            plt.yticks([])
            idx = random.randint(0,len(Y_train) - 1)
            plt.imshow(np.uint8(X_train[idx].reshape(28,28) * 255),cmap = 'gray')
            plt.title(str(Y_train[idx]),color='r')
        plt.show()
    if vector_form:
        return X_train,Y_train,X_test,Y_test
    else:
        return X_train.reshape(60000,1,28,28),Y_train,X_test.reshape(10000,1,28,28),Y_test 
