#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import tensorflow as tf

def orthogonal_initializer(scale=1.0):
    # From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if len(shape) == 1:
            v = np.random.normal(0.0, 1.0, shape)
            return tf.constant(scale * v, dtype=dtype)
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  # this needs to be corrected to float32
        # print('you have initialized one orthogonal matrix.')
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=dtype)
    return _initializer

def get_feed_dict(batch_begin, batch_end, num_gpus, x, y, h, towers):
    feed_dict = dict()
    step = (batch_end - batch_begin) // num_gpus
    gpu_tasks = [[batch_begin + i * step, batch_begin + (i + 1) * step]
                 for i in range(num_gpus)]
    gpu_tasks[-1][-1] = batch_end
    for i, (b, e) in enumerate(gpu_tasks):
        feed_dict[towers[i].x] = x[b: e, :, :, :]
        feed_dict[towers[i].target] = y[b: e, :]
        if h is not None:
            feed_dict[towers[i].h] = h
    return feed_dict

def sum_info(tower_infos):
    sum_infos = []
    for i in range(len(tower_infos[0])):
        expanded_info = []
        for info in tower_infos:
            expanded_info.append(tf.expand_dims(info[i], 0))
        info = tf.concat(expanded_info, 0)
        info = tf.reduce_sum(info, 0)
        sum_infos.append(info)
    return sum_infos

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def fully_connected(model_name, layer_name, var_in, dim_in, dim_out,
                    initializer, transfer):
    with tf.variable_scope(model_name):
        with tf.variable_scope(layer_name):
            W = tf.get_variable("W", [dim_in, dim_out],
                                initializer=initializer)
            b = tf.get_variable("b", [dim_out],
                                initializer=initializer)
    z_hat = tf.matmul(var_in, W)
    z_hat = tf.nn.bias_add(z_hat, b)
    y_hat = transfer(z_hat)
    return W, b, z_hat, y_hat


def crossprop_layer(model_name, layer_name, var_in, dim_in, dim_hidden, dim_out, gate_fun, initializer):
    with tf.variable_scope(model_name):
        with tf.variable_scope(layer_name):
            U = tf.get_variable('U', [dim_in, dim_hidden],
                                initializer=initializer)
            b_hidden = tf.get_variable('b_hidden', [dim_hidden],
                                       initializer=initializer)
            W = tf.get_variable('W', [dim_hidden, dim_out],
                                initializer=initializer)
            b_out = tf.get_variable('b_out', [dim_out],
                                    initializer=initializer)
    net = tf.matmul(var_in, U)
    net = tf.nn.bias_add(net, b_hidden)
    phi = gate_fun(net)
    y = tf.matmul(phi, W)
    y = tf.nn.bias_add(y, b_out)
    return U, b_hidden, net, phi, W, b_out, y


def convolution_2d(model_name, layer_name, var_in, f, dim_in, dim_out,
                   initializer, transfer, stride=1):
    with tf.variable_scope(model_name):
        with tf.variable_scope(layer_name):
            W = tf.get_variable('W', shape=[f, f, dim_in, dim_out],
                                initializer=initializer)
            b = tf.get_variable('b', shape=[dim_out],
                                initializer=initializer)
    z_hat = tf.nn.conv2d(var_in, W, strides=[1, stride, stride, 1], padding="SAME")
    z_hat = tf.nn.bias_add(z_hat, b)
    y_hat = transfer(z_hat)
    return W, b, z_hat, y_hat


class Relu:
    def __init__(self):
        self.gate_fun = tf.nn.relu
        self.gate_fun_gradient = \
            lambda phi, net: tf.where(net >= 0, tf.ones(tf.shape(net)), tf.zeros(tf.shape(net)))


class Tanh:
    def __init__(self):
        self.gate_fun = tf.tanh
        self.gate_fun_gradient = \
            lambda phi, net: tf.subtract(1.0, tf.pow(phi, 2))


class ConvLayers:
    def __init__(self, name, dims, gate, initializer):
        n1, n2, d0, f1, d1, f2, d2 = dims
        self.x = tf.placeholder(tf.float32, shape=(None, n1, n2, d0))  # input
        # layer 1: conv
        W_1, b_1, z_hat_1, r_hat_1 = convolution_2d(
            name, "layer_1", self.x, f1, d0, d1,
            tf.random_normal_initializer(),
            gate.gate_fun)
        # layer 1.5: pool
        s_hat_1 = tf.nn.max_pool(
            r_hat_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # layer 2: conv
        W_2, b_2, z_hat_2, r_hat_2 = convolution_2d(
            name, "layer_2", s_hat_1, f2, d1, d2,
            tf.random_normal_initializer(),
            gate.gate_fun)
        # layer 2.5: pool
        s_hat_2 = tf.nn.max_pool(
            r_hat_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        shape_2 = s_hat_2.get_shape().as_list()
        y_hat_2 = tf.reshape(s_hat_2, [-1, shape_2[1] * shape_2[2] * shape_2[3]])

        self.var_out = y_hat_2
        self.trainable_vars = [W_1, b_1, W_2, b_2]
