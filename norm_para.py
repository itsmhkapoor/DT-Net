"""
Created on May, 2020

author: Mohit Kapoor
mkapoor@student.ethz.ch
"""

import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
# path to dataset for which 98% normalization needs to be performed
path_train = 'path/to/dataset'
N = 5220
def comp_norm(path_train): 
    
    train_x = []
    for x in range(1,N+1):
        x1 = scipy.io.loadmat(f'{path_train}/label_{x}.mat')['out']
        train_x.append(x1)
        print(x)
    train = np.nan_to_num(np.array(train_x, dtype = np.float16))
    amax = np.percentile(train,98)
    amin = np.percentile(train,2)
    return amax, amin

amax, amin = comp_norm(path_train)
print(amax, amin)
