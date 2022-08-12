"""
Created on May, 2020

author: Mohit Kapoor
mkapoor@student.ethz.ch
"""

import tensorflow as tf
import model_unsuper
import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt

batch_size = 15
hm_epochs = 100
lr = 0.000001
N = 5849
N_val = batch_size
normalize = True
image_size = [None, 1329, 253, 1]

path_train = 'path/to/train/images'
path_val = 'path/to/val/images'
path_model = 'path/to/save/model/ckpt'

#Normalization parameters (98%) calculated once using norm_para.py
amin = -0.024139404296875
amax = 0.024139404296875

# here shows you how to relate iterations to epochs
iter_per_epoch = int(math.ceil(N/batch_size))
max_iter = iter_per_epoch * hm_epochs


def ncc(x, y):
    """
    Computes Normalized Cross Correlation between x and y
    """
    mean_x = tf.reduce_mean(x, [1,2,3], keep_dims=True)
    mean_y = tf.reduce_mean(y, [1,2,3], keep_dims=True)
    mean_x2 = tf.reduce_mean(tf.square(x), [1,2,3], keep_dims=True)
    mean_y2 = tf.reduce_mean(tf.square(y), [1,2,3], keep_dims=True)
    stddev_x = tf.reduce_sum(tf.sqrt(
                                    mean_x2 - tf.square(mean_x)), [1,2,3], keep_dims=True)
    stddev_y = tf.reduce_sum(tf.sqrt(
                                    mean_y2 - tf.square(mean_y)), [1,2,3], keep_dims=True)
    return tf.reduce_mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))

def _tf_fspecial_gauss(sizex, sizey, sigma):
    """
    Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-sizex//2 + 1:sizex//2 + 1, -sizey//2 + 1:sizey//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)

def tf_lcc(img1, img2, mean_metric=True, sizex=155, sizey=5, sigma=1.5, deps=1e-4):
    """
    Computes Localized Cross Correlation between img1 and img2
    """
    window = _tf_fspecial_gauss(sizex, sizey, sigma) # window shape [sizex, sizey]
    
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2

    value = sigma12/(tf.sqrt(sigma1_sq + deps)*tf.sqrt(sigma2_sq + deps))
    loss = 1 - value

    if mean_metric:
        loss = tf.reduce_mean(loss)
    return loss

def readImages(path_train): 
    train_x1 = []
    train_x2 = []
    for x in range(1,N+1):
        x1 = scipy.io.loadmat(f'{path_train}/rf1_{x}.mat')['bf_s']
        x2 = scipy.io.loadmat(f'{path_train}/rf2_{x}.mat')['bf_s']
        train_x1.append(x1.reshape(x1.shape[0], x1.shape[1], 1))
        train_x2.append(x2.reshape(x2.shape[0], x2.shape[1], 1))
    return train_x1, train_x2

def readImages_val(path_val): 
    train_x1 = []
    train_x2 = []
    for x in range(1,N_val+1):
        x1 = scipy.io.loadmat(f'{path_val}/val_rf1_{x}.mat')['bf_s']
        x2 = scipy.io.loadmat(f'{path_val}/val_rf2_{x}.mat')['bf_s']
        train_x1.append(x1.reshape(x1.shape[0], x1.shape[1], 1))
        train_x2.append(x2.reshape(x2.shape[0], x2.shape[1], 1))
    return np.array(train_x1), np.array(train_x2)
        

interpolator = model_unsuper.forward(image_size, batch_size)
cost = tf_lcc(interpolator['y1'], interpolator['x1'])
optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(init_op)
    train_x1, train_x2 = readImages(path_train)
    # normalize inputs
    val_x1, val_x2 = readImages_val(path_val)
    val_x1 = (val_x1 - amin)/(amax - amin)
    val_x2 = (val_x2 - amin)/(amax - amin)
    print(val_x1.shape)
    fvals_train = []
    fvals_valid = []
    for step in range(max_iter): #max_iter: maximum number of iterations
       
        i = np.random.randint(N - batch_size)
        # list slicing
        batch_x1 = np.array(train_x1[i:i+batch_size])
        batch_x1 = (batch_x1 - amin)/(amax - amin)
        batch_x2 = np.array(train_x2[i:i+batch_size])
        batch_x2 = (batch_x2 - amin)/(amax - amin)
        _, c = sess.run([optimizer, cost], feed_dict = {interpolator['x1']: batch_x1, interpolator['x2']: batch_x2})

        if (step+1) % 100 == 0: # print and write the error every 100 iterations also save checkpoint every 100 epoch
            print('Iteration', step, 'loss:', c)
            fvals_train.append(c)
            # compute validation loss
            c_valid = sess.run(cost, feed_dict = {interpolator['x1']: val_x1, interpolator['x2']: val_x2})
            print('Iteration', step, 'valid_loss:', c_valid)
            fvals_valid.append(c_valid)
            save_path = saver.save(sess, f'{path_model}/model_{batch_size}_{hm_epochs}.ckpt')

    print('Training done!')
    # Plot Loss curves
    plt.figure()
    plt.plot(fvals_train, label='Training loss')
    plt.plot(fvals_valid, label='Validation loss')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{path_model}/loss_{batch_size}_{hm_epochs}.png', bbox_inches='tight')
    # Save final model
    save_path = saver.save(sess, f'{path_model}/model_{batch_size}_{hm_epochs}.ckpt')
    print("Model saved in path: %s" % save_path)
