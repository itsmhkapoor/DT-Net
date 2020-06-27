"""
Created on May, 2020

author: Mohit Kapoor
mkapoor@student.ethz.ch
"""


import os

import numpy as np
import scipy.io
import tensorflow as tf
import model_new
import sys
from matplotlib import pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

path = '/srv/beegfs02/scratch/sosrecon/data/MS_mel_aug/test_images'
path_results = '/scratch_net/beaker/mkapoor/sem1/mfin-cycle-master/results/test/'

#Normalization parameters (98%) calculated once using norm_para.py
amin = -0.024139404296875
amax = 0.024139404296875

def readImages(path): 
    test_1 = []
    test_2 = []
    x1 = scipy.io.loadmat(f'{path}/test_rf1_32.mat')['bf_s']
    x2 = scipy.io.loadmat(f'{path}/test_rf2_32.mat')['bf_s']
    test_1.append(x1.reshape(x1.shape[0], x1.shape[1], 1))
    test_2.append(x2.reshape(x2.shape[0], x2.shape[1], 1))
    return np.array(test_1), np.array(test_2)


tf.set_random_seed(1.0)


"""
create graph
"""

image_size = [None, 1329, 253, 1]   
interpolator = model_new.forward(image_size, 1)
    
saver = tf.train.Saver(max_to_keep=2)
sess = tf.Session(config=config)

init_from_saved_model = True

if init_from_saved_model:
    saver.restore(sess, "model/anisotropic_lcc/model_15_100.ckpt")
else:
    sess.run(tf.global_variables_initializer())
    
"""
test
"""          

test_1,test_2 = readImages(path)
test_1 = (test_1 - amin)/(amax - amin)
test_2 = (test_2 - amin)/(amax - amin)

h, x2_hat = sess.run([interpolator['h'], interpolator['y1']], \
                     feed_dict={interpolator['x1']: test_1, interpolator['x2']: test_2}) 



_dict = {}
_dict['x2_hat'] = x2_hat
result_name = f'{path_results}rf_flow_32.mat'
flow = h[-1]
flow = np.array(flow[0,:,:,0])
_dict['p'] = flow
scipy.io.savemat(result_name,_dict)

# uncomment to visualize in python (usually visualized in MATLAB along with pseudo ground truth and B-Mode images)
#plt.imshow(flow)
#plt.colorbar()
#plt.savefig(f'{path_results}rf_flow_32.png')