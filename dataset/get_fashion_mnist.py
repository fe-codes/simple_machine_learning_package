import numpy as np
import struct
import random
import matplotlib.pyplot as plt

def get_data(vector_form = False,if_show = False):
    '''
    By default returns X_train.reshape(60000,1,28,28),Y_train,X_test.reshape(10000,1,28,28),Y_test
    if vector_form = True is given, returns X_train,Y_train,X_test,Y_test
    '''
    
##    np.random.seed(1234) # set seed for deterministic ordering
##    p = np.random.permutation(mnist.data.shape[0])

    f = open('E:/dataset/fashion-mnist/train-images-idx3-ubyte','rb')    
    magic = struct.unpack('>i',f.read(4))[0] # > is for big-endian
    n = struct.unpack('>i',f.read(4))[0]
    h = struct.unpack('>i',f.read(4))[0]
    w = struct.unpack('>i',f.read(4))[0]
##    print magic,n,h,w    
    data = struct.unpack('B' * n * h * w,f.read(n * h * w))
    X = np.array(data).reshape(n,h * w)
    X = X.astype(np.float32)/255    
    X_train = X

    f = open('E:/dataset/fashion-mnist/t10k-images-idx3-ubyte','rb')    
    magic = struct.unpack('>i',f.read(4))[0] # > is for big-endian
    n = struct.unpack('>i',f.read(4))[0]
    h = struct.unpack('>i',f.read(4))[0]
    w = struct.unpack('>i',f.read(4))[0]
##    print magic,n,h,w    
    data = struct.unpack('B' * n * h * w,f.read(n * h * w))
    X = np.array(data).reshape(n,h * w)
    X = X.astype(np.float32)/255
    
    X_test = X

    f = open('E:/dataset/fashion-mnist/train-labels-idx1-ubyte','rb')    
    magic = struct.unpack('>i',f.read(4))[0] # > is for big-endian
    n = struct.unpack('>i',f.read(4))[0]
##    print magic,n 
    data = struct.unpack('B' * n ,f.read(n))
    Y = np.array(data).reshape(n,)
    
    Y_train = Y
    f = open('E:/dataset/fashion-mnist/t10k-labels-idx1-ubyte','rb')    
    magic = struct.unpack('>i',f.read(4))[0] # > is for big-endian
    n = struct.unpack('>i',f.read(4))[0]
##    print magic,n 
    data = struct.unpack('B' * n ,f.read(n))
    Y = np.array(data).reshape(n,)
    
    Y_test = Y
    
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
