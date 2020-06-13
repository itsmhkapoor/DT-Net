"""
Created on May, 2020

author: Mohit Kapoor
mkapoor@student.ethz.ch
"""


import os

import numpy as np
import scipy.io
import tensorflow as tf
import model_new_super
import sys
from matplotlib import pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

path = '/srv/beegfs02/scratch/sosrecon/data/test6'
path_labels = '/srv/beegfs02/scratch/sosrecon/data/test6_labels'
path_results = '/scratch_net/beaker/mkapoor/sem1/mfin-cycle-master/results/super/l2/'

N=32
t = 1
def readImages(path): 
    test_1 = []
    test_2 = []
    for x in range(t,N+1):
        x1 = scipy.io.loadmat(f'{path}/test_rf1_{x}.mat')['bf_s']
        x2 = scipy.io.loadmat(f'{path}/test_rf2_{x}.mat')['bf_s']
        test_1.append(x1.reshape(x1.shape[0], x1.shape[1], 1))
        test_2.append(x2.reshape(x2.shape[0], x2.shape[1], 1))
    return np.array(test_1), np.array(test_2)

def readLabels(path_train): 
    train_label = []
    for x in range(1,N+1):
        x1 = scipy.io.loadmat(f'{path_train}/label_{x}.mat')['out']
        train_label.append(x1.reshape(x1.shape[0], x1.shape[1], 1))
    return np.nan_to_num(np.array(train_label))


tf.set_random_seed(1.0)

# Define normalization parameters 98% (middle)
amin = -0.024139404296875
amax = 0.024139404296875
lmax = 2.9453125
lmin = -2.783203125

# Define normalization parameters 98% for 6 channel
#amin = -0.0225067138671875
#amax = 0.0225830078125
#lmax = 1.7041015625
#lmin = -1.6630859375
"""
create graph
"""

image_size = [None, 1329, 253, 1]   
interpolator = model_new_super.forward(image_size)
    
saver = tf.train.Saver(max_to_keep=2)
sess = tf.Session(config=config)

init_from_saved_model = True

if init_from_saved_model:
    saver.restore(sess, "model/supervised6/model_15_100_1e-06_norm.ckpt")
else:
    sess.run(tf.global_variables_initializer())
    
"""
test
"""          

test_1,test_2 = readImages(path)
labels = readLabels(path_labels)
# normalize
test_1 = (test_1 - amin)/(amax - amin)
test_2 = (test_2 - amin)/(amax - amin)


flow = sess.run(interpolator['flow'], \
                     feed_dict={interpolator['x1']: test_1, interpolator['x2']: test_2}) 

# denormalize
flow = (flow)*(lmax-lmin)+(lmin)

# compute RMSE and MAE over test set
rmse = np.sqrt(np.mean(np.square(flow - labels)))
mae = np.mean(np.abs(flow - labels))
print("RMSE:", str(rmse), "MAE:", str(mae))

_dict = {}
result_name = f'{path_results}rf_flow_1.mat'
_dict['p'] = flow
scipy.io.savemat(result_name,_dict)
# uncomment to visualize in python (usually visualized in MATLAB along with pseudo ground truth and B-Mode images)
#plt.imshow(flow)
#plt.colorbar()
#plt.savefig(f'/scratch_net/beaker/mkapoor/sem1/mfin-cycle-master/examples/super/rf/l1/rf_flow_{N}.png')