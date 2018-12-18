import numpy as np
import struct
import random
import matplotlib.pyplot as plt
##from skimage import io
##import cPickle

def get_data(if_show = False):
    '''
    returns X_train,Y_train,X_test,Y_test,Y_c_train,Y_c_test
    '''
    print 'Loading data might take 15 seconds'
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    Y_c_train = []
    Y_c_test = []
    data_path  = 'E:/dataset/cifar/cifar-100-binary/cifar-100-binary/'

    f = open(data_path + 'train.bin','rb')
    for j in range(50000):
        l_c = struct.unpack('b',f.read(1))
        l = struct.unpack('b',f.read(1))
        r = struct.unpack('b'*1024,f.read(1024))
        g = struct.unpack('b'*1024,f.read(1024))
        b = struct.unpack('b'*1024,f.read(1024))
        a=np.zeros((32,32,3))
        a[:,:,0]=np.matrix(r).reshape((32,32))
        a[:,:,1]=np.matrix(g).reshape((32,32))
        a[:,:,2]=np.matrix(b).reshape((32,32))
        Y_c_train.append(l_c)
        Y_train.append(l)
        X_train.append(a)

    f = open(data_path + 'test.bin','rb')
    for j in range(10000):
        l_c = struct.unpack('b',f.read(1))
        l = struct.unpack('b',f.read(1))
        r = struct.unpack('b'*1024,f.read(1024))
        g = struct.unpack('b'*1024,f.read(1024))
        b = struct.unpack('b'*1024,f.read(1024))
        a=np.zeros((32,32,3))
        a[:,:,0]=np.matrix(r).reshape((32,32))
        a[:,:,1]=np.matrix(g).reshape((32,32))
        a[:,:,2]=np.matrix(b).reshape((32,32))
        Y_c_test.append(l_c)
        Y_test.append(l)
        X_test.append(a)
        
    if if_show:
        plt.figure()
        for i in range(100):
            plt.subplot(10,10,i + 1)
            plt.xticks([])
            plt.yticks([])
            idx = random.randint(0,len(Y_train) - 1)
            plt.imshow(np.uint8(X_train[idx]))
            plt.title(str(Y_train[idx]),color='r')
        plt.show()

    X_train = np.uint8(np.array(X_train)).transpose(0,3,1,2)
    X_test = np.uint8(np.array(X_test)).transpose(0,3,1,2)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    Y_c_train = np.array(Y_c_train)
    Y_c_test = np.array(Y_c_test)
    return X_train,Y_train.reshape(50000,),X_test,Y_test.reshape(10000,),Y_c_train.reshape(50000,),Y_c_test.reshape(10000,)
