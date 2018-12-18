import numpy as np
from sklearn.datasets import fetch_mldata
import random
import matplotlib.pyplot as plt

def get_data(input_shape = (1,1,32,32),output_shape = None):
    '''
    This function returns train and test samples satisfying the 
    input shape matches the given input_shape 
    output shape matches the given output_shape, if omitted, output is of shape (1,) * input_shape[0]
    '''
    X_train = np.random.rand(*input_shape)
    X_test = np.random.rand(*input_shape)
    if output_shape is None:
        Y_train = np.array(range(input_shape[0]))
        Y_test = np.array(range(input_shape[0]))
    else:
        Y_train = np.random.rand(*output_shape)
        Y_test = np.random.rand(*output_shape)
    assert(X_train.shape[0] == Y_train.shape[0])    
    return X_train,Y_train,X_test,Y_test
