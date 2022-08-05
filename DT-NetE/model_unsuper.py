"""
Created on May, 2020

author: Mohit Kapoor
mkapoor@student.ethz.ch

Adapted from: https://github.com/linz94/mfin-cycle
author: Lin Zhang
Computer Vision Lab, ETH Zurich
lin.zhang@vision.ee.ethz.ch
"""
"""
References
[1] https://github.com/daviddao/spatial-transformer-tensorflow
[2] https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow
"""

import numpy as np
import tensorflow as tf


def _tf_fspecial_gauss(size, sigma):
    """
    copied from [2]
    """
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def avg_pool_2x1(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  
def weight_variable(shape):
    std = np.sqrt(2/(shape[0]*shape[1]*shape[2]))
    initial = tf.random_normal(shape, stddev=std)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def upsample_layer(input_layer, w, b):
    height = input_layer.shape[1]
    width =  input_layer.shape[2]
    h_upsample = tf.image.resize_images(input_layer, [2*int(height), 2*int(width)])
    h_conv1 = tf.nn.relu(conv2d(h_upsample, w) + b)
    return h_conv1


def initial_weights():
    n_basic_convolution = 7  
    n_filters_convolution = [2,32,64,128,128,64,32,1]
    f_size = [17, 17, 13, 13, 7, 7, 1, 5, 5, 3, 3, 3, 3, 1]
    
    weights = []
    biases = []
    
    for l in range(n_basic_convolution):
        w_size = [f_size[l],f_size[l+7],n_filters_convolution[l],n_filters_convolution[l+1]]

        w = weight_variable(w_size)
        b = bias_variable([n_filters_convolution[l+1]])
        weights.append(w)
        biases.append(b)
            
                    
    return weights, biases


def layer_evaluation(input_layer, weights, biases):
    h = []
    n_convolution = 3
    n_deconvolution = 2
    h_current = input_layer
    
    for l in range(n_convolution):
        if l==0:
            w1 = weights[0]
            b1 = biases[0]
            w2 = weights[1]
            b2 = biases[1]
        elif l==1:
            w1 = weights[2]
            b1 = biases[2]
            w2 = weights[3]
            b2 = biases[3]
        else:
            w1 = weights[4]
            b1 = biases[4]
            w2 = weights[5]
            b2 = biases[5]
        
        h1 = tf.nn.relu(conv2d(h_current, w1) + b1)
        h.append(h1)
        h2 = tf.nn.relu(conv2d(h1, w2) + b2)
        h.append(h2)
        if l==2:
            h_current = avg_pool_2x2(h2)
            h.append(h_current)
        else:
            h_current = avg_pool_2x1(h2)
            h.append(h_current)

    h_p1 = tf.add(tf.nn.conv2d(h_current, weights[6], strides=[1, 1, 1, 1], padding='SAME'), biases[6])
    h.append(h_p1)
    
    return h


def grid_generator(p, mode):

    num_batch = tf.shape(p)[0]
    
    x = tf.linspace(0.0, 252.0, 253)
    y = tf.linspace(0.0, 1328.0, 1329)
    
    if mode == 'xy':
        x_t, y_t = tf.meshgrid(x, y)
    elif mode == 'ij':
        x_t, y_t = tf.meshgrid(x, y, indexing = 'ij')
    
    sampling_grid = tf.stack([x_t, y_t], axis=-1)
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1, 1]))
    
    sampling_grid = tf.add(sampling_grid, p)

    return sampling_grid


def get_pixel_value(img, x, y, mode):
    
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    batch_idx = tf.cast(batch_idx, 'float32')
    b = tf.tile(batch_idx, (1, height, width))
    b = tf.cast(b, 'int32')

    if mode == 'xy':
        indices = tf.stack([b, y, x], 3)
    elif mode == 'ij':
        indices = tf.stack([b, x, y], 3)

    return tf.gather_nd(img, indices)


def bilinear_interpolator(img, p, mode):
    """
    adapted from [1]
    """
    
    max_y = tf.cast(1329 - 1, 'int32')
    max_x = tf.cast(253 - 1, 'int32')
    zero = tf.zeros([], dtype=tf.int32)
    
    sampling_grid = grid_generator(p, mode)
    x = sampling_grid[:,:,:,0]
    y = sampling_grid[:,:,:,1]
    
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    

    Ia = get_pixel_value(img, x0, y0, mode)
    Ib = get_pixel_value(img, x0, y1, mode)
    Ic = get_pixel_value(img, x1, y0, mode)
    Id = get_pixel_value(img, x1, y1, mode)
         
    
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')
    
    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)
    
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    
    return out

def compute_grad(im):
    grad_dy = im[:,1:,:,:] - im[:,:-1,:,:]
    grad_dx = im[:,:,1:,:] - im[:,:,:-1,:]
    return grad_dx, grad_dy

def compute_smooth_loss(flowU, flowV):
    flow_gradU_dx, flow_gradU_dy = compute_grad(flowU)
    flow_gradV_dx, flow_gradV_dy = compute_grad(flowV)
    loss = tf.reduce_mean(tf.abs(flow_gradU_dx)) + tf.reduce_mean(tf.abs(flow_gradU_dy)) \
         +  tf.reduce_mean(tf.abs(flow_gradV_dx)) + tf.reduce_mean(tf.abs(flow_gradV_dy))
    return loss


def tranformation_composition(p1, p2):
    p_new1 = bilinear_interpolator(tf.expand_dims(p2[:,:,:,0], axis=3), p1, 'ij')
    p_new2 = bilinear_interpolator(tf.expand_dims(p2[:,:,:,1], axis=3), p1, 'ij')    
    return tf.concat([p_new1, p_new2], axis=-1) + p1


def compute_consistency_loss(p1, p2):
    cost_x = tf.reduce_mean(tf.square(p1[:,:,:,0] - p2[:,:,:,0]))
    cost_y = tf.reduce_mean(tf.square(p1[:,:,:,1] - p2[:,:,:,1]))
    return tf.add(cost_x, cost_y)


def forward(input_shape, batch_size):

    x1 = tf.placeholder(tf.float32, input_shape, name='x1')
    x2 = tf.placeholder(tf.float32, input_shape, name='x2')
                 
    x = tf.concat([x1, x2], 3)

    
    w, b = initial_weights()
        
    h = layer_evaluation(x,w,b)
    
    flow = h[-1]
    #New part for only axial flow
    flow_up = tf.image.resize(flow, [1329,253], 'bilinear')
    v = tf.zeros([batch_size,1329,253,1])
    flow_all = tf.concat([flow_up, v], axis=-1)

    x2_interp = bilinear_interpolator(x2, flow_all, 'xy')
    
    flowU = tf.expand_dims(flow_all[:,:,:,0], axis=3)
    flowV = tf.expand_dims(flow_all[:,:,:,1], axis=3)
    
    smooth_loss = compute_smooth_loss(flowU, flowV)
    
    
           
    return {'x1': x1, 'x2': x2, 'y1': x2_interp, 'h' : h, 'W' : w, 'smooth' : smooth_loss}

        